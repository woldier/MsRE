<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">
MsRE: Towards Efficient Remote Sensing Segmentation via Vision Foundation Models
</h2>

Bin Wang <sup>1</sup>,
Shun Lv <sup>1</sup>,
Zhi Li <sup>1</sup>,
Fei Deng <sup>2</sup>,
Yiguang Liu<sup>1†</sup>

<sup>1</sup> Sichuan University
<sup>2</sup> Chengdu University of Technology

<sup>†</sup> Corresponding author.

[//]: # (<sup>*</sup> Equal contribution.)

<div style="display: flex; align-items: center; justify-content: center;">
<p align="center">
  <br align="center">
    <a href='https://ieeexplore.ieee.org/document/11599658'><img src='http://img.shields.io/badge/TGRS-2026.3711219-006699.svg?logo=IEEE&logoColor=006699'></a>
    <img alt="Static Badge" src="https://img.shields.io/badge/python-v3.8-green?logo=python">
    <img alt="Static Badge" src="https://img.shields.io/badge/torch-v2.1.2-B31B1B?logo=pytorch">
    <img alt="Static Badge" src="https://img.shields.io/badge/torchvision-v0.16.2-B31B1B?logo=pytorch">
    <img alt="Static Badge" src="https://img.shields.io/badge/mmengine-v0.9.1-blue">
    </br>
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues/woldier/MsRE">
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues-closed/woldier/MsRE?color=ab7df8">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/woldier/MsRE?style=flat&color=red">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/woldier/MsRE?style=flat&color=af2626">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=BinWang.MsRE&left_color=%2363C7E6&right_color=%23CEE75F">
</p>
</div>
<br/>
<img src="figs/overview.png" alt="MsRE" style="max-width: 100%; height: auto;"/>
<div style="display: flex; align-items: center; justify-content: center;"> Network Overview </div>

</div>


### 🔍️🔍️ NEWS
- [2026/07/14] 🤗🤗 The `Training Code` has been updated.
- [2026/07/06] 🎉🎉 Our paper is accepted by IEEE TGRS ([Link](https://ieeexplore.ieee.org/document/11599658))!
- [2026/05/04] 🧨🧨 Update project page && readme.md.
- [2026/04/13] ✨✨ Init Repo.

[//]: # (- [2025/11/17] ✨✨ The [arxiv] paper will coming soon.)

## 📖 Clone Repo

---
We add [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) as our repository 
[submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) .

So, one should clone this repository use the script as follows:

<details>
<summary>clone repository</summary>

```shell
git clone --recurse-submodules https://github.com/woldier/MsRE


```
> ### Tips
> If one already cloned the project and forgot --recurse-submodules,
> 
> ```shell
>  # cloned the project and forgot clone submodules 🥲🥲
>  git clone https://github.com/woldier/Bridge 
> 
>  # initialize and update each submodule in the repository 🥰🥰
>  git submodule update --init
>  ```
> 

</details>


after that, we link `mmsegmentation/mmseg` $\to$ `mmseg`:

<details>
<summary>soft link</summary>

```shell
$ pwd work_dir/MsRE
ln -s mmsegmentation/mmseg mmseg
```
</details>

## 🛠️️ 1. Creating Virtual Environment

This repo use `python-3.8`, for `nvcc -v` with `cuda >= 11.6`.

`torch 2.1.1`, `cuda 12.1`, `mmcv 2.1.0`, `mmengine 0.9.1`

<details>
<summary>Install script</summary>


```shell
conda create -n  MsRE  python==3.8 -y
conda activate MsRE


pip install torch==2.1.2+cu121  torchvision==0.16.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
# for CN user use follow script
pip install torch==2.1.2+cu121  torchvision==0.16.2+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/  

pip install mmcv==2.1.0 mmengine==0.9.1 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

pip install -r submodule-mmseg/requirements/runtime.txt
```
</details>

Installation of the reference document refer:

Torch and torchvision versions relationship.

[![Official Repo](https://img.shields.io/badge/Pytorch-vision_refer-EE4C2C?logo=pytorch)](https://github.com/pytorch/vision#installation)
[![CSDN](https://img.shields.io/badge/CSDN-vision_refer-FC5531?logo=csdn)](https://blog.csdn.net/shiwanghualuo/article/details/122860521)


## 📂 2.Preparation of datasets


We selected Postsdam, Vaihingen and LoveDA as benchmark datasets.

### 2.1 Download of datasets

### ISPRS Potsdam

The [Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The '2_Ortho_RGB.zip' and '5_Labels_all_noBoundary.zip' are required.



### ISPRS Vaihingen


The [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Vaihingen.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip' are required.



#### LoveDA

The data could be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing).

Or it can be downloaded from [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF), you should run the following command:

<details>
<summary> loveda download</summary>

```shell

cd /{your_project_base_path}/Bridge/data/LoveDA

# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```
</details>

### 2.2 Data set preprocessing
Place the downloaded file in the corresponding path
The format is as follows:

<details>
<summary>file structure</summary>

```text
MsRE/
├── data/
│   ├── LoveDA/
│   │   ├── Test.zip
│   │   ├── Train.zip
│   │   └── Val.zip
├── ├── Potsdam_RGB/
│   │   ├── 2_Ortho_RGB.zip
│   │   └── 5_Labels_all_noBoundary.zip
├── ├── Vaihingen_IRRG/
│   │   ├── ISPRS_semantic_labeling_Vaihingen.zip
│   │   └── ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip
```
</details>

after that we can convert dataset:

<details>
<summary>details</summary>

- Potsdam
```shell
python tools/convert_datasets/potsdam.py data/Potsdam_RGB/ --clip_size 512 --stride_size 512 -o data/potsdam
```
- Vaihingen
```shell
python tools/convert_datasets/vaihingen.py data/Vaihingen_IRRG/ --clip_size 512 --stride_size 256 -o data/vaihingen
```


- LoveDA
```shell
python tools/convert_datasets/loveda.py data/LoveDA/ --tmp_dir data -o data/loveda
```

</details>

## 🔥 3. Training

<details>
<summary>training scripts</summary>

```bash
# loveda
python tools/train.py configs/msre/msre_dinov3_vit-large_segmentor_1xb2-amp-40k_loveda-512x512.py

# potsdam
python tools/train.py configs/msre/msre_dinov3_vit-large_segmentor_1xb2-amp-40k_potsdam-512x512.py

# vaihingen
python tools/train.py configs/msre/msre_dinov3_vit-large_segmentor_1xb2-amp-40k_vaihingen-512x512.py
```
</details>


## Acknowledgements
This project is built upon [OpenMMLab](https://openmmlab.com/codebase). We thank the OpenMMLab developers.



## Citation
If you use Geoad in your research, please cite:
```bibtex
        @ARTICLE{11599658,
          author={Wang, Bin and Lv, Shun and Li, Zhi and Deng, Fei and Liu, Yiguang},
          journal={IEEE Transactions on Geoscience and Remote Sensing},
          title={MsRE: Towards Efficient Remote Sensing Segmentation via Vision Foundation Models},
          year={2026},
          volume={},
          number={},
          pages={1-1},
          keywords={Modeling;Remote sensing;Semantic segmentation;Training;Tuning;Decoding;LoRa;Visualization;Vegetation;Head;Vision Foundation Models;Semantic Segmentation;Parameter-Efficient Fine-Tuning;Remote Sensing},
          doi={10.1109/TGRS.2026.3711219}
        }
```
