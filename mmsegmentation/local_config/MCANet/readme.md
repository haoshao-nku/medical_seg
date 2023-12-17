# <p align=center>`MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention`</p>

> **Authors:**
> [Hao Shao](https://scholar.google.com/citations?hl=en&user=vB4DPYgAAAAJ), [Quansheng Zeng](), [Qibin Hou](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=en&oi=ao), &[Jufeng Yang](https://scholar.google.com/citations?user=c5vDJv0AAAAJ&hl=en&oi=ao).



This document mainly contains [MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention](https://arxiv.org/abs/2312.08866)'s training, testing and other experimental methods, experimental results and visualization results, etc.

## **Abstract**

Efficiently capturing multi-scale information and building long-range dependencies among pixels are essential for medical image segmentation because of the various sizes and shapes of the lesion regions or organs. In this paper, we present Multi-scale Cross-axis Attention (MCA) to solve the above challenging issues based on the efficient axial attention. Instead of simply connecting axial attention along the horizontal and vertical directions sequentially, we propose to calculate dual cross attentions between two parallel axial attentions to capture global information better. To process the significant variations of lesion regions or organs in individual sizes and shapes, we also use multiple convolutions of strip-shape kernels with different kernel sizes in each axial attention path to improve the efficiency of the proposed MCA in encoding spatial information. We build the proposed MCA upon the MSCAN backbone, yielding our network, termed MCANet. Our MCANet with only 4M+ parameters performs even better than most previous works with heavy backbones (e.g., Swin Transformer) on four challenging tasks, including skin lesion segmentation, nuclei segmentation, abdominal multi-organ segmentation, and polyp segmentation.

## Architecture


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

## Experiments

### Change dataset path

- 1.Download the [Polypseg](https://github.com/DengPingFan/PraNet) dataset, then decompress the dataset.
- 2.Update the training path and test path of **/medical_seg/mmsegmentation/local_config/_base_/datasets/polypseg.py** in the project, on lines 55, 56, 67, and 68 respectively.
> We recommend using absolute paths instead of relative paths when updating paths of dataset. ISIC2018, DSB2018 and Synapse datasets are also changed in the above way.

> It should be noted that when using the synapse dataset, you first need to convert the dataset. The conversion method can be found in /medical_seg/mmsegmentation/tools/dataset_converters/synapse.py.

### Training
Please confirm whether you are currently under the mmsegmentation directory. If not, please enter the mmsegmentation directory. Then run the following code in terminal:

- python tools/train.py /medical_seg/mmsegmentation/local_config/MCANet/main/MCANet_mscsn_t_polypseg_512*512_80k.py
- python tools/train.py /medical_seg/mmsegmentation/local_config/MCANet/main/MCANet_mscan_t_synapse_512*512_50k.py
- **......**



> During training, verification is performed every 8,000 iterations, and the checkpoint file is saved at the same time. Batch size and validation set evaluation metrics can be changed in the corresponding configuration files.

### Testing

> You can find the directory with the same name as the configuration file in the **/medical_seg/mmsegmentation/work_dirs** folder. Stored below are logs, checkpoints and other files generated when testing with the current configuration.

 The command to test the model is as follows:

- python tools/test.py  /medical_seg/mmsegmentation/local_config/MCANet/main/MCANet_mscsn_t_polypseg_512*512_80k.py  /medical_seg/mmsegmentation/work_dirs/MCANet_mscsn_t_polypseg_512*512_80k/iter_80000.pth --eval mIoU

>  You can replace iter_80000.pth to evaluate the performance of different checkpoints. Similarly, you can replace mIoU and use different evaluation indicators to evaluate the model.

> The evaluation indicators supported by mmsegmentation can be found in **/medical_seg/mmsegmentation/mmseg/evaluation/metrics**.

### Calculate the Flops and Parameters
Please run the following command:
- python /medical_seg/mmsegmentation/local_config/MCANet/main/MCANet_mscsn_t_polypseg_512*512_80k.py --shape 512 512

> You can calculate it by replacing "512 512" with the image size you want.
> You can replace the configuration files to evaluate flops and parameters for different networks.

### Ablation Study

If you want to perform ablation experiments, we recommend that you refer to the methods in /medical_seg/mmsegmentation/local_config/Polyper-AAAI2024/readme.md.

> You can also refer to [mmsegmentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/) of "自定义组件/新增模块" to make changes. We implement it based on mmsegmentation. It is recommended that you read [mmsegmentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/) for a better understanding.