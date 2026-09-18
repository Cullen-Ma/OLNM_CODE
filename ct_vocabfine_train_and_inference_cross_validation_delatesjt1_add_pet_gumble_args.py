import os
import torch
from sklearn.model_selection import KFold
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from data_inference import CTReportDatasetinfer
from sklearn.model_selection import StratifiedKFold

from transformer_maskgit import CTViT

from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
import torch.nn.functional as F
from src.args import parse_arguments
from src.models.utils import cosine_lr, torch_load, LabelSmoothing
import pandas as pd
from zero_shot_our_crossval import CTClipInference

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_lr(optimizer):
    # Function to get the current learning rate of the optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']

def finetune(args):
    # Initialize BERT tokenizer and text encoder
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))

    # ==================== 新增/修改: 将新参数传递给CTViT ====================
    # Initialize image encoder and clip model
    image_encoder = CTViT(
        dim=512, 
        codebook_size=8192, 
        image_size=480, 
        patch_size=20,
        temporal_patch_size=10, 
        spatial_depth=4, 
        temporal_depth=4,
        dim_head=32, 
        heads=8,
        # 新增的双流模型参数
        num_local_enhancement_tokens=args.num_local_enhancement_tokens,
        local_region_size=args.local_region_size,
        local_transformer_depth=args.local_transformer_depth,
        gumbel_tau=args.gumbel_tau
    )
    # =================================================================

    clip = CTCLIP(
        image_encoder=image_encoder, 
        text_encoder=text_encoder,
        dim_image=512, 
        dim_text=768, 
        dim_latent=512,
        extra_latent_projection=False, 
        use_mlm=False,
        downsample_image_embeds=False, 
        use_all_token_embeds=False
    )

    print('Fine-tuning end-to-end')
    
    ds = CTReportDatasetinfer(
        data_folder=args.data_folder, 
        pet_data_folder=args.pet_data_folder,
        csv_file=args.reports_file, 
        labels=args.labels
    )

    k_folds = 8
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    df = pd.read_csv(args.labels)
    labels_array = df['Lymphadenopathy'].values

    print(f"Labels array size: {len(labels_array)}")
    assert len(ds) == len(labels_array), "Dataset and labels array must have the me length."


    for fold, (train_ids, val_ids) in enumerate(skf.split(torch.zeros(len(labels_array)), labels_array)):
        clip.load(args.pretrained)
        
        num_classes = 18
        
        model = clip

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        print("Layer-wise Parameters:")
        print("-" * 60)
        for name, param in model.named_parameters():
            param_count = param.numel()
        print("-" * 60)

        print(f'FOLD {fold}')
        print('--------------------------------')
        import pdb
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(ds, num_workers=8, batch_size=1, sampler=train_subsampler)
        validation_loader = DataLoader(ds, num_workers=6, batch_size=1, sampler=val_subsampler)
        num_batches = len(train_loader)
        model.cuda()
        devices = list(range(torch.cuda.device_count()))
        print('Using devices', devices)
        model = torch.nn.DataParallel(model, device_ids=devices)
        model.train()

        loss_fn = torch.nn.MSELoss()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
        
        
        print('Start training ...')
        print(f'Total epoch: {args.epochs}')
        print(f'Save every: {args.save_every}')
        print(f"num_batches: {num_batches}")

        for epoch in range(args.epochs):
            for i, batch in tqdm.tqdm(enumerate(train_loader)):
                start_time = time.time()                
                step = i + epoch * num_batches
                scheduler(step)

                (inputs_ct, inputs_pet), input_text, labels, _ = batch

                logits = []
                labels_tensor = labels.float().to(torch.device('cuda'))
                optimizer.zero_grad()

                logits_list = []
                labels_list = []

                pathologies = ['Lymphadenopathy']

                for l in range(len(labels_tensor)):
                    text_yes = ""
                    text_no = ""
                    if labels_tensor[l] == 1:
                        text_yes = text_yes + f"{pathologies[l]} is present. "
                        text_no = text_no + f"{pathologies[l]} is not present. "
                    if labels_tensor[l] == 0:
                        text_yes = text_yes + f"{pathologies[l]} is not present. "
                        text_no = text_no + f"{pathologies[l]} is present. "

                    input_text_str = str(input_text[0])

                    if input_text_str[-1] == "." or input_text_str[-1] == ",":
                        text_yes_add_finding_impression = input_text_str + text_yes
                        text_no_add_finding_impression = input_text_str + text_no
                    else:
                        text_yes_add_finding_impression = input_text_str + "." + text_yes
                        text_no_add_finding_impression = input_text_str + "." + text_no
                    text = [text_yes_add_finding_impression, text_no_add_finding_impression]
                    
                    text_tokens = tokenizer(
                        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(
                        torch.device('cuda'))
                    
                    output = model(text_tokens, inputs_ct, inputs_pet) 

                    logits = F.softmax(output, dim=0)
                    labels = torch.tensor([1.0, 0.0]).cuda()
                    logits_list.append(logits)
                    labels_list.append(labels)

                concat_logits = torch.cat(logits_list, dim=0)
                concat_labels = torch.cat(labels_list, dim=0)

                loss = loss_fn(concat_logits, concat_labels)
                loss.backward()
                optimizer.step()

                # print(get_lr(optimizer))

                batch_time = time.time() - start_time

                if i % args.print_every == 0:
                    percent_complete = 100 * i / len(train_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                        f"Loss: {loss.item():.6f}\tBatch (t) {batch_time:.3f}", flush=True
                    )
            inference = CTClipInference(
                model,
                validation_loader=validation_loader,
                batch_size = 1,
                results_folder=f"{args.result_folder}_{fold}",
                num_train_steps = 1,
            )

            inference.infer(epoch)
            
            if epoch % args.save_every == 0:
                os.makedirs(args.save, exist_ok=True)

                model_to_save = model.module if hasattr(model, 'module') else model

                model_path = os.path.join(args.save, f'epoch_{epoch+1}.pt')
                print('Saving model to', model_path)

                torch.save(model_to_save.state_dict(), model_path)

if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)