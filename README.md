# UATST
Code for UATST

## Requirements  
We recommend the following configurations:  
- python 3.8
- PyTorch 1.12.0
- CUDA 12.2
- pip install ftfy regex tqdm
- conda install -c anaconda git
- pip install git+https://github.com/openai/CLIP.git

## Model Training  
- Download the content dataset: [MS-COCO](https://cocodataset.org/#download).
- Run the following command:
```
python train.py --content_dir /data/train2014 --text artemis_dataset_release_v0.csv
```

## Model Testing
- Run the following command:
```
python Eval.py --content content/ --text "A fauvism style painting with vibrant colors"
```
