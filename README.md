<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">
MsRE: Towards Efficient Remote Sensing Segmentation via Vision Foundation Models
</h2>

<div style="display: flex; align-items: center; justify-content: center;">
<p align="center">
  <br align="center">
    <a href='#'><img src='http://img.shields.io/badge/Paper-arxiv.xxx.xxx-B31B1B.svg?logo=arXiv&logoColor=B31B1B'></a>
    <img alt="Static Badge" src="https://img.shields.io/badge/python-v3.8-green?logo=python">
    <img alt="Static Badge" src="https://img.shields.io/badge/torch-v1.0.2-B31B1B?logo=pytorch">
    <img alt="Static Badge" src="https://img.shields.io/badge/mmcv-v1.5.0-blue">
    <img alt="Static Badge" src="https://img.shields.io/badge/torchvision-v0.11.3-B31B1B?logo=pytorch">
    </br>
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues/woldier/MsRE">
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues-closed/woldier/MsRE?color=ab7df8">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/woldier/MsRE?style=flat&color=red">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/woldier/MsRE?style=flat&color=af2626">
</p>
</div>
<br/>
<img src="figs/overview.png" alt="MsRE" style="max-width: 100%; height: auto;"/>
<div style="display: flex; align-items: center; justify-content: center;"> Network Overview </div>

</div>


### 🔍️🔍️ NEWS

- [2026/04/13] ✨✨ Init Repo.

[//]: # (- [2025/11/17] ✨✨ The [arxiv] paper will coming soon.)

## 1. Creating Virtual Environment

---

<details>
<summary>Install script</summary>

```shell
pip install torch==1.10.2+cu111 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu111/?mirror_intel_list
pip install torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install kornia matplotlib prettytable timm yapf==0.40.1
```

for CN user:
```shell
pip install torch==1.10.2+cu111 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu111/?mirror_intel_list
pip install torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install kornia matplotlib prettytable timm yapf==0.40.1
```
</details>

Installation of the reference document refer:

Torch and torchvision versions relationship.

[![Official Repo](https://img.shields.io/badge/Pytorch-vision_refer-EE4C2C?logo=pytorch)](https://github.com/pytorch/vision#installation)
[![CSDN](https://img.shields.io/badge/CSDN-vision_refer-FC5531?logo=csdn)](https://blog.csdn.net/shiwanghualuo/article/details/122860521)

Version relationship of mmcv and torch.

[![MMCV](https://img.shields.io/badge/mmcv-vision_refer-blue)](https://mmcv.readthedocs.io/zh-cn/v1.5.0/get_started/installation.html)

## Acknowledgements
This project is built upon [OpenMMLab](https://openmmlab.com/codebase). We thank the OpenMMLab developers.

## Citation
If you use Geoad in your research, please cite:
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```