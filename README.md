# <p align=center>`Medical Image Segmentation`</p>


:fire::fire: This is an official repository of our work on medical image segmentation:fire::fire:

> If you have any questions about our work, feel free to contact me via e-mail (shaoh@mail.nankai.edu.cn).


## 🌤️ Highlights
- (2023.12.16) The code of [MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention](https://arxiv.org/abs/2312.08866) release.
- (2023.12.16) The code of [Polyper: Boundary Sensitive Polyp Segmentation](https://arxiv.org/abs/2312.08735) release.
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
pip install einops
```

The following methods can be used to verify that the experimental environment is successfully set up:
```
1. mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
2. python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```
After the preceding two steps are successfully run, if the result.png file is generated under the mmsegmentation folder, the environment is successfully created.The result.png as shown in the following.

<p align="center"><img width="800" alt="image" src="https://github.com/haoshao-nku/medical_seg/blob/master/mmsegmentation/demo/result.jpg"></p> 

**1. Dataset**
> The dataset used in the experiment can be obtained in the following methods:
- For polyp segmentation task: [Polypseg](https://github.com/DengPingFan/PraNet): including Kvasir, - CVC-ClinicDB, CVC-ColonDB, EndoScene and ETIS dataset.
- For abdominal multi-organ segmentation task: [Synapse](https://github.com/Beckschen/TransUNet).
- For skin lesion segmentation task: [ISIC-2018](https://challenge.isic-archive.com/data/#2018).
- For nuclei segmentation task: [DSB2018](https://www.kaggle.com/c/data-science-bowl-2018).

**2. Experiments**
We recommend that you place the project folder in a location such as a solid state drive, and put the checkpoint files generated from the experiment on a mechanical hard drive to save space, so you can choose to create a soft connection. Specific practices are as follows:

> ln -s   "mechanical hard disk path"  /medical_seg/mmsegmentation/work_dirs

If your hardware resources are relatively rich, ignore this advice.

> **Note: Our experiment is implemented based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). The environment configuration can also refer to the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), and questions about the entire project can refer to the [official documentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/).**

## Our Work

### [Polyper: Boundary Sensitive Polyp Segmentation](https://arxiv.org/abs/2312.08735) AAAI 2024

#### **Abstract**

We present a new boundary sensitive framework for polyp segmentation, called Polyper. Our method is motivated by a clinical approach that seasoned medical practitioners often leverage the inherent features of interior polyp regions to tackle blurred boundaries. Inspired by this, we propose explicitly leveraging polyp regions to bolster the model’s boundary discrimination capability while minimizing computation. Our approach first extracts boundary and polyp regions from the initial segmentation map through morphological operators. Then, we design the boundary sensitive attention that concentrates on augmenting the features near the boundary regions using the interior polyp regions’s characteristics to generate good segmentation results. Our proposed method can be seamlessly integrated with classical encoder networks, like ResNet-50, MiT-B1, and Swin Transformer. To evaluate the effectiveness of Polyper, we conduct experiments on five publicly available challenging datasets, and receive state-of-the-art performance on all of them.

#### Architecture

<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/pipline_polyper.png"/> <br />
    <em> 
    Figure 1: Overall architecture of Polyper. We use the Swin-T from Swin Transformer as the encoder. The decoder is divided into two main stages. The first potential boundary extraction (PBE) stage aims to capture multi-scale features from the encoder, which are then aggregated to generate the initial segmentation results. Next, we extract the predicted polyps' potential boundary and interior regions using morphology operators. In the second boundary sensitive refinement (BSR) stage, we model the relationships between the potential boundary and interior regions to generate better segmentation results.
    </em>
</p>


<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/refine_polyper.png"/> <br />
    <em> 
    Figure 2: Detailed structure of boundary sensitive attention (BSA) module. This process is separated into two parallel branches, which systematically capitalize on the distinctive attributes of polyps at various growth stages, both in terms of spatial and channel characteristics. `B' and `M' indicate the number of pixels in the boundary and interior polyp regions within an input of size H*W and C channels.
    </em>
</p>

#### Experiments

> For training, testing and other details can be found at **/medical_seg/mmsegmentation/local_config/Polyper-AAAI2024/readme.md**.

### [MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention](https://arxiv.org/abs/2312.08866)

#### **Abstract**


Efficiently capturing multi-scale information and building long-range dependencies among pixels are essential for medical image segmentation because of the various sizes and shapes of the lesion regions or organs. In this paper, we present Multi-scale Cross-axis Attention (MCA) to solve the above challenging issues based on the efficient axial attention. Instead of simply connecting axial attention along the horizontal and vertical directions sequentially, we propose to calculate dual cross attentions between two parallel axial attentions to capture global information better. To process the significant variations of lesion regions or organs in individual sizes and shapes, we also use multiple convolutions of strip-shape kernels with different kernel sizes in each axial attention path to improve the efficiency of the proposed MCA in encoding spatial information. We build the proposed MCA upon the MSCAN backbone, yielding our network, termed MCANet. Our MCANet with only 4M+ parameters performs even better than most previous works with heavy backbones (e.g., Swin Transformer) on four challenging tasks, including skin lesion segmentation, nuclei segmentation, abdominal multi-organ segmentation, and polyp segmentation.

#### Architecture



<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/pipeline-MCANet.png"/> <br />
    <em> 
    Figure 1: Overall architecture of the proposed MCANet. We take the MSCAN network proposed in SegNeXt as our encoder because of its capability of capturing multi-scale features. The feature maps from the last three stages of the encoder are combined via upsampling and then concatenated as the input of the decoder. Our decoder is based on multi-scale cross-axis attention, which takes advantage of both multi-scale convolutional features and the axial attention.
    </em>
</p>



<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/decoder-MCANet.png"/> <br />
    <em> 
    Figure 2: Detailed structure of the proposed multi-scale cross-axis attention decoder. Our decoder contains two parallel paths, each of which contains multi-scale 1D convolutions and cross-axis attention to aggregate the spatial information. Note that we do not add any activation functions in decoder.
    </em>
</p>


#### Experiments

> For training, testing and other details can be found at **/medical_seg/mmsegmentation/local_config/MCANet/readme.md**.


## Acknowlegement

Thanks [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) providing a friendly codebase for segmentation tasks. And our code is built based on it.

## Reference
You may want to cite:
```

```

### License

Code in this repo is for non-commercial use only.
