import numbers
from einops import rearrange
import cv2 as cv
import numpy as np
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from ..utils import resize

def erosion_to_dilate(output):
        z = output.cpu().detach().numpy()
        z = np.where(z > 0.3, 1.0, 0.0)  #covert segmentation result
        z = torch.tensor(z)    
        kernel = np.ones((4, 4), np.uint8)   # kernal matrix
        maskd = np.zeros_like(output.cpu().detach().numpy())  #result array
        maske = np.zeros_like(output.cpu().detach().numpy())  #result array
        for i in range(output.shape[0]):
            y = z[i].permute(1,2,0)
            erosion = y.cpu().detach().numpy()
            dilate = y.cpu().detach().numpy()
            dilate = np.array(dilate,dtype='uint8')
            erosion = np.array(erosion,dtype='uint8')
            erosion = cv.erode(erosion, kernel, 4)  
            dilate = cv.dilate(dilate, kernel, 4)
            mask1 = torch.tensor(dilate-erosion).unsqueeze(-1).permute(2,0,1)
            mask2 = torch.tensor(erosion).unsqueeze(-1).permute(2,0,1)
            maskd[i] = mask1
            maske[i] = mask2
        maskd = torch.tensor(maskd)
        maskd = maskd.cuda()
        maske = torch.tensor(maske)
        maske = maske.cuda()        
        return maskd,maske

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)    

class RefineAttention(nn.Module):
    def __init__(self, dim, num_heads,LayerNorm_type,):
        super(RefineAttention, self).__init__()
        self.num_heads = num_heads   
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)   
    def forward(self, x, mask_d,mask_e):
        b,c,h,w = x.shape   
        x = self.norm(x)
        y1 = x*(1-mask_e)
        y2 = x*(1-mask_d)
        out_sa = x.clone()
        with torch.no_grad():
            for i in range(b):
                z_d = []
                z_e = []
                pos_d = np.argwhere(mask_d[i][0].cpu().detach().numpy() == 1)
                pos_e = np.argwhere(mask_e[i][0].cpu().detach().numpy() == 1)            
                for j in range(c):
                    z_d.append(x[i,j,pos_d[:,0],pos_d[:,1]])
                    z_e.append(x[i,j,pos_e[:,0],pos_e[:,1]])
                #spatial att
                z_d = torch.stack(z_d)
                z_e = torch.stack(z_e)
                z_e = z_e.cuda()
                z_d = z_d.cuda()
                k1 = rearrange(z_e, '(head c) z -> head z c', head=self.num_heads)
                v1 = rearrange(z_e, '(head c) z -> head z c', head=self.num_heads)
                q1 = rearrange(z_d, '(head c) z -> head z c', head=self.num_heads)    
                q1 = torch.nn.functional.normalize(q1, dim=-1)
                k1 = torch.nn.functional.normalize(k1, dim=-1)   
                # attn1 = torch.einsum('hzc,hzc->hzz', q1, k1)
                attn1 = (q1 @ k1.transpose(-2, -1))
                attn1 = attn1.softmax(dim=-1) 
                out1 = (attn1 @ v1) + q1  
                # out1 = torch.einsum('hzz,hzc->hzc', attn1, v1) +q1   
                out1 = rearrange(out1, 'head z c -> (head c) z', head=self.num_heads)
                for j in range(c):
                    out_sa[i,j,pos_d[:,0],pos_d[:,1]] = out1[j]      
        # channel att
        k2 = rearrange(y2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(y2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(y1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)    
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)      
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)   
        out2 = (attn2 @ v2) + q2      
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)   
        out = x + out_sa + out2

        return out
@MODELS.register_module()
class Polyper(BaseDecodeHead):
    def __init__(self,in_channels,image_size,heads,
                 **kwargs):
        super(Polyper, self).__init__(in_channels,input_transform = 'multiple_select',**kwargs)
        self.image_size = image_size

        self.refine1 = RefineAttention(in_channels[0],heads,LayerNorm_type = 'WithBias')
        self.refine2 = RefineAttention(in_channels[1],heads,LayerNorm_type = 'WithBias')
        self.refine3 = RefineAttention(in_channels[2],heads,LayerNorm_type = 'WithBias')
        self.refine4 = RefineAttention(in_channels[3],heads,LayerNorm_type = 'WithBias')
        self.align1 = ConvModule(
            in_channels[3],
            in_channels[2],
            1)
        self.align2 = ConvModule(
            in_channels[2],
            in_channels[1],
            1)        
        self.align3 = ConvModule(
            in_channels[1],
            in_channels[0],
            1 )

    def forward(self, inputs):
        inputs = [resize(
                level,
                size=self.image_size,
                mode='bilinear'
            ) for level in inputs]        

        #stage4
        y3 = inputs[3]
        #stage3
        conv_y1 = self.align1(y3)
        y2 = inputs[2]+conv_y1
        #stage3
        conv_y2 = self.align2(y2)
        y1 = inputs[1]+conv_y2           
        #stage3
        conv_y3 = self.align3(y1)
        y0 = inputs[0]+conv_y3
        mask_d,mask_e = erosion_to_dilate(self.cls_seg(y0))
        #stage4
        y3 = self.refine4(y3, mask_d,mask_e)
        #stage3
        conv_y1 = self.align1(y3)
        y2 = self.refine3(y2+conv_y1,mask_d,mask_e)
        #stage3
        conv_y2 = self.align2(y2)
        y1 = self.refine2(y1+conv_y2,mask_d,mask_e)            
        #stage3
        conv_y3 = self.align3(y1)
        y0 = self.refine1(y0+conv_y3,mask_d,mask_e)

        output = self.cls_seg(y0)
        return output