import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm
import pdb

class CTReportDatasetinfer(Dataset):
    # 新增/修改: __init__函数增加了 pet_data_folder 参数
    def __init__(self, data_folder, pet_data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, labels = "labels.csv"):
        self.data_folder = data_folder
        # 新增/修改: 存储PET数据文件夹的路径
        self.pet_data_folder = pet_data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        # 新增/修改: 为CT和PET分别创建数据到张量的转换函数
        self.ct_nii_to_tensor = partial(self.ct_nii_img_to_tensor, transform = self.transform)
        self.pet_nii_to_tensor = partial(self.pet_nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text

    # 新增/修改: 重写 prepare_samples 以同时处理 CT 和 PET
    def prepare_samples(self):
        samples = []
        # glob.glob现在只查找CT文件
        ct_patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        print("Preparing CT/PET sample pairs...")
        for patient_folder in tqdm.tqdm(ct_patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                ct_nii_files = glob.glob(os.path.join(accession_folder, '*.npz'))

                for ct_nii_file in ct_nii_files:
                    accession_number_npz = os.path.basename(ct_nii_file)
                    accession_number_nii = accession_number_npz.replace(".npz", ".nii.gz")

                    if accession_number_nii not in self.accession_to_text:
                        continue

                    # --- 寻找对应的PET文件 ---
                    # 1. 获取CT文件相对于其基准目录的相对路径
                    relative_path = os.path.relpath(ct_nii_file, self.data_folder)
                    # 2. 用这个相对路径在PET基准目录中构建绝对路径
                    pet_nii_file = os.path.join(self.pet_data_folder, relative_path)

                    # 3. 只有当对应的PET文件也存在时，才将其视为一个有效的样本对
                    if not os.path.exists(pet_nii_file):
                        # print(f"Warning: Corresponding PET file not found for {ct_nii_file}. Skipping.")
                        continue
                    
                    impression_text = self.accession_to_text[accession_number_nii]
                    text_final = ""
                    for text in list(impression_text):
                        text = str(text)
                        if text == "Not given.":
                            text = ""
                        text_final = text_final + text

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number_nii]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        # 新增/修改:样本中现在包含CT和PET两个文件的路径
                        samples.append((ct_nii_file, pet_nii_file, text_final, onehotlabels[0]))
                        self.paths.append(ct_nii_file) # 保持 self.paths 不变，如果需要的话
        
        print(f"Found {len(samples)} valid CT/PET pairs.")
        return samples

    def __len__(self):
        return len(self.samples)

    # 新增/修改: 将原函数重命名为 ct_nii_img_to_tensor，专用于CT处理
    def ct_nii_img_to_tensor(self, path, transform):
        img_data = np.load(path, allow_pickle=True)['arr_0']
        
        # --- CT特有的预处理 (HU值窗口化) ---
        #img_data= np.transpose(img_data, (1, 2, 0))
        img_data = img_data*1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (((img_data+400 ) / 600)).astype(np.float32)
        # --- CT预处理结束 ---

        # --- 共享的空间变换 ---
        tensor = torch.tensor(img_data)
        target_shape = (480,480,240)
        h, w, d = tensor.shape

        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        # --- 空间变换结束 ---

        return tensor

    # 新增/修改: 增加一个专用于PET数据预处理的新函数
    def pet_nii_img_to_tensor(self, path, transform):
        img_data = np.load(path, allow_pickle=True)['arr_0']

        # --- PET特有的预处理 (SUV值标准化) ---
        # PET值 (SUV) 通常是正数。我们裁剪掉极端离群值并归一化。
        suv_max = 10.0  # 一个常用的临床上限，可以根据需要调整
        img_data = np.clip(img_data, 0, suv_max)
        # 归一化到 [0, 1] 范围
        img_data = (img_data / suv_max).astype(np.float32)
        # --- PET预处理结束 ---

        # --- 共享的空间变换 (与CT完全相同以保证对齐) ---
        tensor = torch.tensor(img_data)
        target_shape = (480,480,240)
        h, w, d = tensor.shape

        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=0) # PET用0填充背景
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        # --- 空间变换结束 ---

        return tensor

    # 新增/修改: __getitem__现在返回一个包含CT和PET张量的元组
    def __getitem__(self, index):
        # 从样本列表中解包出 CT 和 PET 两个文件路径
        ct_nii_file, pet_nii_file, input_text, onehotlabels = self.samples[index]

        # 分别为CT和PET文件生成张量
        ct_video_tensor = self.ct_nii_to_tensor(ct_nii_file)
        pet_video_tensor = self.pet_nii_to_tensor(pet_nii_file)
        
        # 清理文本
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  
        
        # 使用CT文件名来获取accession number
        name_acc = ct_nii_file.split("/")[-2]
        
        # 返回一个包含两个张量的元组作为第一项
        return (ct_video_tensor, pet_video_tensor), input_text, onehotlabels, name_acc