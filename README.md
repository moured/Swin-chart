<h1 align="center">📊 Swin-Chart </h1>
<h3 align="center">Swin-chart: An efficient approach for chart classification</h3>
<h4 align="center">Volume 185, September 2024, Pages 203-209</h4>
<h3 align="center"> Maintained by <a href="https://www.linkedin.com/in/omar-moured/">🔗 Omar Moured</a> | <a href="https://github.com/moured"> 💻 GitHub</a> </h3>

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S0167865524002447">
    <img src="https://img.shields.io/badge/ScienceDirect-Paper-orange?logo=Elsevier" /></a>
<!--     <a href="https://www.sciencedirect.com/science/article/pii/S0167865524002447">
    <img src="https://img.shields.io/badge/PDF-arXiv-brightgreen" /></a> -->
<!--     <a href="https://yufanchen96.github.io/projects/RoDLA/">
    <img src="https://img.shields.io/badge/Project-Homepage-red" /></a> -->
    <a href="https://pytorch.org/get-started/previous-versions/#linux-and-windows">
    <img src="https://img.shields.io/badge/Framework-PyTorch%202.1.1-orange" /></a> 
    <a href="https://github.com/yufanchen96/RoDLA/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
    <img alt="visits" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmoured%2FSwin-chart&count_bg=%23DC00FF&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits&edge_flat=false">
</p>

## 🏡 About

Swin-Chart is a Swin Transformer-based model for chart classification across various datasets. It achieved first place in the chart classification task at the ICDAR 2023 CHART Infographics competition.
    
## 🔎 Introduction

We introduce **Swin-Chart**, a Swin Transformer-based model with finetuning and a greedy weight-averaging strategy for chart image classification. The pre-trained Swin Transformer captures both local and global feature dependencies, and after finetuning on chart image datasets, the best model weights are averaged to achieve optimal performance. Experiments on five chart image datasets demonstrate that Swin-Chart outperforms state-of-the-art models across all datasets.
<p align="center">
    <img src="assets/swinchart_model.jpg" width="860" />
</p>

## 📝 Catalog
- [x] Training Code
- [x] Testing Code
- [ ] Swin-Chart Checkpoints
  - [x] Best Swin-Base Checkpoint 
  - [ ] Greedy Soup Checkpoint
- [ ] Checkpoint Averaging Code

## 📦 Installation
**1. Clone the repository**
```
git clone [https://github.com/yufanchen96/RoDLA.git](https://github.com/moured/Swin-chart.git)
cd Swin-chart
```

**2. Create a conda virtual environment**
```
# create virtual environment
conda create -n swinchart python=3.8 -y
conda activate swinchart
```

**3. Install Dependencies**
```
pip install -r requirements.txt 
```

## 📂 Dataset Preparation

Download the [UB-PMC dataset](https://www.kaggle.com/datasets/pranithchowdary/icpr-2022).

Prepare the dataset for by splitting it into 80% for training and 20% for testing. Modify the dataset paths in [ICPR_split.sh](./ICPR_split.sh), then execute the following command:
```
bash ICPR_split.sh
```

## 🚀 Quick Start

### Training
**1. Modify the train/test dataset split paths and model selection in [config.json](./config.json), You can choose between baseline or large models.**
```
"model_name": "swin_base_patch4_window7_224"
```
OR
```
"model_name": "swin_large_patch4_window7_224"
```

**2. Start Training**
```
python train.py
```

### Evaluate the Swin-Chart model
You can download our best model from the [Releases](https://github.com/moured/Swin-chart/releases/tag/swin-base-best) page.

**1. Modify the test dataset path and model checkpoint in [test.sh](./test.sh)**

**2. Start Testing**
```
bash test.sh
```

## 🌳 Citation
If you find this code useful for your research, please consider citing:
```
@article{DHOTE2024203,
title = {Swin-chart: An efficient approach for chart classification},
journal = {Pattern Recognition Letters},
volume = {185},
pages = {203-209},
year = {2024},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2024.08.012},
url = {https://www.sciencedirect.com/science/article/pii/S0167865524002447},
author = {Anurag Dhote and Mohammed Javed and David S. Doermann},
keywords = {Chart classification, Swin transformer, Deep learning, Scientific documents},
}
```
