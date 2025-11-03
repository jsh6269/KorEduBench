# Development Settings

## Install Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Prepare Dataset

Download [Curriculum-Level Subject Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM010&aihubDataSe=data&dataSetSn=71855)  
Note that we only use texts (not the images) which means **label directory** of dataset above is used in our project.

## Preprocess Dataset

```bash
cd src/preprocessing

# extract unique standards from dataset
python extract_standards.py {Train_label_path}
python add_text_to_standards.py {Train_label_path} --max_texts 160

# 빈 데이터가 중간에 삽입되어 있는지 확인하는 코드 (not necessary to run)
python verify_not_empty.py {text_achievement_standards.csv_path}

# add text samples to each standards
python check_insufficient_text.py --min_texts 160
python add_additional_text_to_standards.py {Val_label_path} --max_texts 160
python check_insufficient_text.py --min_texts 160

# split csv by subject
python filter_standards.py --num_texts 160
python split_subject.py --input {train.csv_path} --output "train_80"
python split_subject.py --input {val.csv_path} --output "val_80"
```

## Naive Cosine Similarity

```bash
cd cosine_similarity

# example csv_path: ../dataset/subject_text20/과학.csv"
python eval_cosine_similarity.py --input_csv {csv_path}

# calculate every csv files in the folder
python batch_cosine_similarity.py --folder_path {folder_path}
```

## Rerank with Cross Encoder

```bash
# Note that you could give encoding option (default model save path = "./cross_finetuned")
python finetune_cross_encoder.py --input_csv {csv_path}
python eval_cross_encoder.py --input_csv {csv_path} 
```
