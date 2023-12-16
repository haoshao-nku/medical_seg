# <p align=center>`Medical Image Segmentation`</p>


:fire::fire: This is a official repository of our work on medical image segmentation:fire::fire:

> If you have any questions about our work, feel free to contact me via e-mail (shaoh@mail.nankai.edu.cn).


## 🌤️ Highlights
- (2023.12.15) Our paper "**Polyper: Boundary Sensitive Polyp Segmentation**" was accepted by AAAI2024, We have released article on [arXiv](https://arxiv.org/abs/2312.08735).
- (2023.12.15) We have released article on arXiv: [MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention](https://arxiv.org/abs/2312.08866).

## Get Start
> Our experiments are based on ubuntu, and windows is not recommended.
> 
**0. Install**

```
conda create --name medical_seg python=3.8 -y
conda activate medical_seg
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

cd mmsegmentation
pip install -v -e .
pip install ftfy
pip install regex
```

```
The following methods can be used to verify that the experimental environment is successfully set up:

1. mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
2. python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg

After the above two steps are successfully run, a result.png file is generated under the mmsegmentation folder.
<p align="center"><img width="800" alt="image" src="https://github.com/haoshao-nku/medical_seg/blob/master/mmsegmentation/demo/result.jpg"></p> 
```

**1. Dataset**