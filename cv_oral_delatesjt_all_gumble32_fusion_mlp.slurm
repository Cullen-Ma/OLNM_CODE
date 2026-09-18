#!/bin/bash
#SBATCH --job-name=mlp_concat                  # 任务名
#SBATCH --partition=gpu                  # cpu or gpu
#SBATCH --nodelist=gpu02
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=2       
#SBATCH --output=/share/home/kma/ct_clip/CT-CLIP-main_debug_gumble/out_log/out_cv_gumble32_double_flow_mlp_concat.log
#SBATCH --gres=gpu:1


source activate januspro
sleep 10s
python -u /share/home/kma/ct_clip/CT-CLIP-main_debug_gumble/scripts/ct_vocabfine_train_and_inference_cross_validation_delatesjt1_add_pet_gumble_args.py --lr 5e-6 --wd 0.1 --epochs 40 --warmup_length 200 --save-every 10 --save "/share/home/kma/ct_clip/CT-CLIP-main_debug_gumble/output_basepublic_5e6_gumble32_lt6_doubleflow_mlp_concat" --pretrained '/share/home/kma/ct_clip/CT-CLIP-main/output_oral_public_ct_cv_1012/epoch_21.pt' --data-folder "/share/home/kma/ct_clip/CTRATE/dataset/preprocess_data/mask_cut_ct_1009_preprocessed" --pet-data-folder "/share/home/kma/ct_clip/CTRATE/dataset/preprocess_data/pet_oral_delate_shijitan_1009_preprocessed" --reports-file "/share/home/kma/ct_clip/CTRATE/dataset/radiology_text_reports/oral_all_reports_delatesjt_rewrite_0919.csv" --labels "/share/home/kma/ct_clip/CTRATE/dataset/multi_abnormality_labels/oral_all_labels_delate_shijitan.csv"  --num-local-enhancement-tokens 32  --local-transformer-depth 6 --fusion-model "mlp_concat" --result-folder "/share/home/kma/ct_clip/inference_result/data_all_cv/inference_gumble32_5e6_fusion_mlp"
