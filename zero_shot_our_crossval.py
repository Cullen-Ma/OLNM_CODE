from pathlib import Path
from shutil import rmtree
from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

# ==================== 修改 1: 确保导入所有需要的指标计算函数 ====================
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    multilabel_confusion_matrix, 
    f1_score, 
    accuracy_score, 
    roc_auc_score,   # 用于计算 AUC
    recall_score     # 用于计算 Sensitivity/Recall
)
# =========================================================================

import pdb
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from data_inference import CTReportDatasetinfer
import numpy as np
import tqdm
import pandas as pd
import nibabel as nib 

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP

# ... (中间的 helper functions 保持不变: tensor_to_nifti, exists, noop, cycle, yes_or_no, accum_log, apply_softmax, CosineAnnealingWarmUpRestarts) ...

def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    tensor = tensor.cpu()
    if tensor.dim() == 4:
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def apply_softmax(array):
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))
        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma

class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        num_train_steps,
        batch_size,
        validation_loader,
        lr = 1e-4,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 100,
        save_model_every = 2000,
        results_folder = './results',
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr
        self.ds = validation_loader

        # prepare with accelerator
        self.dl_iter=cycle(validation_loader)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optim,
                                                  T_0=4000000,
                                                  T_warmup=10000,
                                                  eta_max=lr)

        (
 			self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return
        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)
        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self, epoch):
        device = self.device
        steps = int(self.steps.item())

        # logs
        logs = {}

        if True:
            with torch.no_grad():
                models_to_evaluate = ((self.CTClip, str(steps)),)

                for model, filename in models_to_evaluate:
                    model.eval()
                    predictedall=[]
                    realall=[]
                    logits = []

                    text_latent_list = []
                    image_latent_list = []
                    accession_names=[]
                    pathologies  = ['Lymphadenopathy']
                    
                    for i in tqdm.tqdm(range(len(self.ds))):
                        valid_data, text, onehotlabels, acc_name = next(self.dl_iter)
                        
                        valid_data_ct, valid_data_pet = valid_data
                        valid_data_ct = valid_data_ct.to(device)
                        valid_data_pet = valid_data_pet.to(device)

                        plotdir = self.result_folder_txt
                        Path(plotdir).mkdir(parents=True, exist_ok=True)

                        predictedlabels=[]
                        onehotlabels_append=[]

                        for pathology in pathologies:
                            text_yes = f"{pathologies[0]} is present. "
                            text_no = f"{pathologies[0]} is not present. "
                            input_text_str = str(text[0])

                            if input_text_str[-1] == "." or input_text_str[-1] == ",":
                                text_yes_add_finding_impression = input_text_str + text_yes
                                text_no_add_finding_impression = input_text_str + text_no
                            else:
                                text_yes_add_finding_impression = input_text_str + "." + text_yes
                                text_no_add_finding_impression = input_text_str + "." + text_no
                            
                            text_prompts = [text_yes_add_finding_impression, text_no_add_finding_impression]
                            
                            text_tokens=self.tokenizer(
                                            text_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                            
                            output = model(text_tokens, valid_data_ct, valid_data_pet)
                            output = apply_softmax(output)

                            append_out=output.detach().cpu().numpy()
                            # append_out[0] is the probability of the "present" class
                            predictedlabels.append(append_out[0])

                        predictedall.append(predictedlabels)
                        realall.append(onehotlabels.detach().cpu().numpy()[0])
                        accession_names.append(acc_name[0])

                    realall = np.array(realall) # Shape: (N, num_pathologies)
                    predictedall = np.array(predictedall) # Shape: (N, num_pathologies)

                    np.savez(f"{plotdir}labels_weights.npz", data=realall)
                    np.savez(f"{plotdir}predicted_weights.npz", data=predictedall)
                    
                    with open(f"{plotdir}accessions_{epoch}.txt", "w") as file:
                        for item in accession_names:
                            file.write(item + "\n")

                    # ==================== 修改 2: 显式计算所有指标 ====================
                    print(f"Calculating comprehensive metrics for Epoch {epoch}...")
                    
                    metrics_list = []
                    
                    # 遍历每一个病理类型进行计算
                    for idx, pathology_name in enumerate(pathologies):
                        y_true = realall[:, idx]
                        y_prob = predictedall[:, idx]
                        
                        # 1. AUC
                        try:
                            val_auc = roc_auc_score(y_true, y_prob)
                        except ValueError:
                            val_auc = 0.5 # 只有一类时处理
                        
                        # 2. 生成预测类别 (阈值 0.5)
                        y_pred = (y_prob >= 0.5).astype(int)
                        
                        # 3. Accuracy
                        val_acc = accuracy_score(y_true, y_pred)
                        
                        # 4. F1-Score
                        val_f1 = f1_score(y_true, y_pred, zero_division=0)
                        
                        # 5. Sensitivity (Recall) & Specificity
                        # confusion_matrix 返回 [[TN, FP], [FN, TP]]
                        if len(np.unique(y_true)) > 1:
                            cm = confusion_matrix(y_true, y_pred)
                            tn, fp, fn, tp = cm.ravel()
                            
                            val_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        else:
                            # 只有正例或只有负例的边缘情况
                            if y_true[0] == 1: # 全是正例
                                val_sens = recall_score(y_true, y_pred)
                                val_spec = 0.0 
                            else: # 全是负例
                                val_sens = 0.0
                                val_spec = recall_score(y_true, y_pred, pos_label=0)

                        metrics_list.append({
                            "Pathology": pathology_name,
                            "AUC": val_auc,
                            "Accuracy": val_acc,
                            "Sensitivity": val_sens,
                            "Specificity": val_spec,
                            "F1_Score": val_f1
                        })
                    
                    # 保存为 Excel
                    df_metrics = pd.DataFrame(metrics_list)
                    
                    # 你可以选择覆盖原来的 aurocs_{epoch}.xlsx 或新建一个文件
                    # 这里我新建一个文件名，确保不会破坏你原来的 evaluate_internal 逻辑
                    output_excel_path = f'{plotdir}comprehensive_metrics_{epoch}.xlsx'
                    writer = pd.ExcelWriter(output_excel_path, engine='xlsxwriter')
                    df_metrics.to_excel(writer, sheet_name='Sheet1', index=False)
                    writer.close()
                    
                    print(f"Metrics saved to {output_excel_path}")
                    # =================================================================

                    # 保留原来的调用，以防 evaluate_internal 里面有绘图逻辑 (plot_roc等)
                    dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)
                    # 如果不需要原来的简版 Excel，可以注释掉下面这几行
                    writer_orig = pd.ExcelWriter(f'{plotdir}aurocs_{epoch}.xlsx', engine='xlsxwriter')
                    dfs.to_excel(writer_orig, sheet_name='Sheet1', index=False)
                    writer_orig.close()
                    
        self.steps += 1
        return logs

    def infer(self, epoch, log_fn=noop):
        device = next(self.CTClip.parameters()).device
        device = self.accelerator.device
        
        while self.steps < self.num_train_steps:
            logs = self.train_step(epoch)
            log_fn(logs)

        self.print('Inference complete')