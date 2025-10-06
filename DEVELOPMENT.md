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
cd dataset

# extract unique standards from dataset
python extract_standards.py {label_path}

# add text samples to each standards
python add_text_to_standards.py {label_path} --max_texts 20

# split csv by subject
python split_subject.py --input {csv_path} --max-texts 20
```

## Naive Cosine Similarity

```bash
cd cosine_similarity

# example csv_path: ../dataset/subject_text20/과학.csv"
python eval_cosine_similarity.py --input_csv {csv_path}"
```
