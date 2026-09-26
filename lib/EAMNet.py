import timm
from re import X
from tkinter.ttk import Scale
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from lib.ConvNeXt import *
from lib.Res2Net_v1b import *
from lib.resnet import *
from lib.RCAB import *
from lib.vgg import vgg16

eps = 1e-12

# ---  可学习的边缘提取模块 ---
class LearnableEdgeModule(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16, out_channels=1):
        super(LearnableEdgeModule, self).__init__()
        # 使用一个小型编码器结构来提取边缘特征
        self.encoder = nn.Sequential(
            BasicConv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
        )
        # 输出层，生成单通道的边缘预测图
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 直接从原始RGB图像中学习边缘
        features = self.encoder(x)
        edge_pred = self.out_conv(features)
        return edge_pred
        
# ---  Sobel 边缘分支模块 ---
class EdgeBranch(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16, out_channels=1):
        super(EdgeBranch, self).__init__()
        
        # 1. 将 RGB 图像转为灰度图
        self.gray_conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # 使用标准的 RGB to Grayscale 权重，并设为不可训练
        rgb_to_gray_weight = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.gray_conv.weight = nn.Parameter(rgb_to_gray_weight, requires_grad=False)
        
        # 2. 定义 Sobel 算子 (不可训练)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.sobel_conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_conv_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_conv_y.weight = nn.Parameter(sobel_y, requires_grad=False)
        
        # 3. 后续处理卷积，用于提取特征和生成最终的边缘图
        self.后续处理 = nn.Sequential(
            BasicConv2d(2, mid_channels, kernel_size=3, stride=1, padding=1), # 输入是 x, y 两个方向的梯度图拼接
            BasicConv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0) # 输出单通道的边缘预测
        )

    def forward(self, x):
        # 传入原始的 RGB 图像 x
        gray_img = self.gray_conv(x)
        
        grad_x = self.sobel_conv_x(gray_img)
        grad_y = self.sobel_conv_y(gray_img)
        
        # 可以选择 L1/L2 范数或者直接拼接
        # edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2) # L2 范数
        edge_combined = torch.cat([torch.abs(grad_x), torch.abs(grad_y)], dim=1) # 拼接
        
        edge_pred = self.后续处理(edge_combined)
        
        return edge_pred

class MBConvBlock(nn.Module):
    """ 轻量级的 MobileNetV2/V3 逆残差块 """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class FFN(nn.Module):
    """ 卷积前馈网络 ConvFFN """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# --- [新增] SE 通道注意力模块 ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
# --- 主模块：HIFM ---
class HIFM(nn.Module):
    def __init__(self, in_channels, embed_dim, out_channels, num_heads=4, token_reduction_scale=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction_scale = token_reduction_scale

        # 1. Proj & Align
        self.proj_x2 = nn.Conv2d(in_channels, embed_dim, 1)
        self.proj_fuse1 = nn.Conv2d(in_channels, embed_dim, 1)
        self.norm_x2 = nn.BatchNorm2d(embed_dim)
        self.norm_fuse1 = nn.BatchNorm2d(embed_dim)
        
        # 2. CNN Local Path (借鉴 CMT/MaxViT 的 Conv)
        self.local_cnn_path = nn.Sequential(
            MBConvBlock(embed_dim, embed_dim, expand_ratio=4),
            MBConvBlock(embed_dim, embed_dim, expand_ratio=4)
        )

        # 3. Token Reduction (借鉴 CMT)
        self.token_reducer = nn.Conv2d(embed_dim, embed_dim, kernel_size=token_reduction_scale, stride=token_reduction_scale)
        self.token_norm = nn.LayerNorm(embed_dim)

        # 4. Cross-Attention Core (核心交互)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.se_layer = SELayer(embed_dim * 2, reduction=8) # 输入通道是 cat 后的 embed_dim * 2

        # 5. Fusion & Gating (融合策略)
        self.fusion_gate_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 1)
        )

        # 6. FFN & Output Projection
        self.ffn = FFN(embed_dim, embed_dim * 4, embed_dim)
        self.norm_ffn = nn.BatchNorm2d(embed_dim)
        self.output_proj = nn.Conv2d(embed_dim, out_channels, 1)

    def forward(self, x2_t, fuse_1, edge_guidance_map):
        #  自动对齐输入特征图的空间分辨率
        if x2_t.shape[2:] != fuse_1.shape[2:]:
            fuse_1 = F.interpolate(fuse_1, size=x2_t.shape[2:], mode='bilinear', align_corners=True)

        B, C, H, W = x2_t.shape
        
        # 1. Proj & Align
        proj_x2 = self.norm_x2(self.proj_x2(x2_t))
        proj_fuse1 = self.norm_fuse1(self.proj_fuse1(fuse_1))
        residual = proj_x2 + proj_fuse1

        # 2. CNN Local Path
        local_features = self.local_cnn_path(proj_x2)

        # 3. Token Reduction
        tokens_x2 = self.token_reducer(proj_x2).flatten(2).transpose(1, 2)
        tokens_fuse1 = self.token_reducer(proj_fuse1).flatten(2).transpose(1, 2)
        tokens_x2 = self.token_norm(tokens_x2)
        tokens_fuse1 = self.token_norm(tokens_fuse1)

        # 4. Cross-Attention Core
        enhanced_tokens_x2, _ = self.cross_attn(query=tokens_x2, key=tokens_fuse1, value=tokens_fuse1)
        enhanced_tokens_fuse1, _ = self.cross_attn(query=tokens_fuse1, key=tokens_x2, value=tokens_x2)
        attn_tokens = enhanced_tokens_x2 + enhanced_tokens_fuse1
        
        attn_out_reduced = attn_tokens.transpose(1, 2).view(B, self.embed_dim, H // self.reduction_scale, W // self.reduction_scale)
        attn_out = F.interpolate(attn_out_reduced, size=(H, W), mode='bilinear', align_corners=False)

        # 5. Fusion & Gating
        edge_gate = F.interpolate(edge_guidance_map, size=(H, W), mode='bilinear', align_corners=False)
        edge_gate = torch.sigmoid(edge_gate) # 确保范围在 (0, 1)

        concatenated_features = torch.cat([local_features, attn_out], dim=1)
        refined_features = self.se_layer(concatenated_features)
        fusion_gate_offset = torch.sigmoid(self.fusion_gate_conv(refined_features))

        final_gate = (1 - edge_gate) * fusion_gate_offset

        fused_features = final_gate * attn_out + (1 - final_gate) * local_features  
        # 6. FFN & Residual
        ffn_out = self.ffn(fused_features)
        output = self.norm_ffn(ffn_out + residual)
        output = self.output_proj(output)

        return output, final_gate

#Channel Reduce
class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel,RFB = False):
        super(Reduction, self).__init__()
        #self.dyConv = Dynamic_conv2d(in_channel,out_channel,3,padding = 1)
        if(RFB):
            self.reduce = nn.Sequential(
                RFB_modified(in_channel,out_channel),
            )
        else:
            self.reduce = nn.Sequential(
                BasicConv2d(in_channel, out_channel, 1),
            )
    def forward(self, x):
        return self.reduce(x)


#
class SEA(nn.Module):
    def __init__(self, channel = 64):
        super(SEA, self).__init__() 
    
        self.upconv2 = conv_upsample(channel=channel)

        self.conv1 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        self.conv2 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        self.conv3 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        self.conv4 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        #self.Bconv1 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        #self.ms = MS_CAM()  
        self.fuse1  = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        self.fuse2  = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))

        self.final_fuse = nn.Sequential(
            ConvBR(channel*2, channel*2,kernel_size=3, stride=1,padding=1),
            BasicConv2d(channel*2, channel,kernel_size=3, stride=1,padding=1),
        )
        
    def forward(self, sen_f, edge_f, edge_previous):   # x guide y
        s1 = F.upsample(sen_f, size=edge_f.size()[2:], mode='bilinear', align_corners=True)
        s1 = self.conv1(s1)  #upsample
        s2 = self.conv2(sen_f) 
        e1 = self.conv3(edge_f)
        e2 = self.conv4(edge_previous)

        e1 = self.fuse1(e1 * s1) + e1
        e2 = self.fuse2(e2 * s2) + e2

        e2  = self.upconv2(e2,e1)  #upsample 

        out  = self.final_fuse(torch.cat((e1,e2),1))

        return out


class Guide_flow(nn.Module):
    def __init__(self, x_channel, y_channel):
        super(Guide_flow, self).__init__()
        self.guidemap = nn.Conv2d(x_channel, 1, 1)

        self.gateconv = GatedSpatailConv2d(y_channel)

    def forward(self, x, y):   # x guide y

        guide = self.guidemap(x)
        guide_flow = F.interpolate(guide, size = y.size()[2:], mode='bilinear') 
        y = self.gateconv(y,guide_flow)

        return y

class GatedSpatailConv2d(nn.Module):
    def __init__(self,channels = 32,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias_attr=False):
        super(GatedSpatailConv2d, self).__init__()
        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(channels+1),
            nn.Conv2d(channels +1, channels +1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels +1, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.conv = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0,dilation=dilation,groups=groups)

    def forward(self, input_features, gating_features):
        cat = torch.cat((input_features, gating_features),1)
        attention = self._gate_conv(cat)
        x = input_features * (attention + 1)
        x = self.conv(x)
        return x


class IntegralAttention (nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(IntegralAttention, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat = nn.Sequential(
            BasicConv2d(in_channel*3, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),

        )
        self.conv_res = BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1)

        self.eps = 1e-5   
        self.IterAtt = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 8, kernel_size=1),
            nn.LayerNorm([out_channel // 8, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // 8, out_channel, kernel_size=1)
        )
        self.ConvOut = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1), 
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        fuse = self.relu(x_cat + self.conv_res(x))

        # can change to the MS-CAM or SE Attention, refer to lib.RCAB.
        context = (fuse.pow(2).sum((2,3), keepdim=True) + self.eps).pow(0.5) # [B, C, 1, 1]
        channel_add_term = self.IterAtt(context)
        out = channel_add_term * fuse + fuse

        out = self.ConvOut(out)

        return out

# EIA module
class EIA(nn.Module):
    def __init__(self, s_channel = 64, h_channel= 64 ,e_channel= 64 ):
        super(EIA, self).__init__()
        #self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv_1 =  ConvBR(h_channel, h_channel,kernel_size=3, stride=1,padding=1)
        self.conv_2 =  ConvBR(s_channel, h_channel,kernel_size=3, stride=1,padding=1)
        self.conv_3 =  ConvBR(e_channel, e_channel,kernel_size=3, stride=1,padding=1)
        self.conv_d1 = BasicConv2d(h_channel, h_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = BasicConv2d(h_channel, h_channel, kernel_size=3, stride=1, padding=1)

        self.attention = IntegralAttention(in_channel=s_channel, out_channel=s_channel)
        self.convfuse1 = nn.Sequential(
            BasicConv2d(s_channel*3, s_channel,kernel_size=3, stride=1,padding=1), 
            BasicConv2d(s_channel, s_channel,kernel_size=3, stride=1,padding=1),
        )



    def forward(self, left, down,edge):
        left_1 = self.conv_1(left)
        down_1 = self.conv_2(down)
        #edge_1 = self.conv_3(edge)

        down_2 = self.conv_d1(down_1)
        left_2 = self.conv_l(left_1)

    #z1 conv(down) * left
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear')
        z1 = F.relu(left_1 * down_2, inplace=True)

    #z2 conv(left) * down
        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')
        z2 = F.relu(down_1 * left_2, inplace=True)

        fuse = self.convfuse1(torch.cat((z1, z2, edge), 1))
    #fuse and Integrity enhence

        out = self.attention(fuse)

        return out

class Network(nn.Module):
    #  ---- VGG16 Backbone ----
       # self.backbone = eval(vgg16)(pretrained = True)
       # enc_channels=[64, 128, 256, 512, 512]
    #
    #  ---- ConvNext Backbone ----
        #self.backbone = convnext_tiny(pretrained=True)
        #enc_channels=[96, 192, 384,768]

    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()

        # ---- Backbone: PVTv2-B4 ----
        self.backbone = timm.create_model('pvt_v2_b4', pretrained=False, features_only=True)
        
        # --- 权重加载逻辑 ---
        if imagenet_pretrained:
            weights_path = './snapshot/EAMNet_E_b4_3/Net_epoch_best.pth'
            state_dict = torch.load(weights_path, map_location='cpu')

            # --- 全新的、精确的 Key 修复逻辑 ---
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('head.'):
                    continue  # 忽略分类头

                new_key = k
                if k.startswith('patch_embed'):
                    if 'patch_embed1.' in k:
                        new_key = k.replace('patch_embed1.', 'patch_embed.')
                    elif 'patch_embed2.' in k:
                        new_key = k.replace('patch_embed2.', 'stages_1.downsample.')
                    elif 'patch_embed3.' in k:
                        new_key = k.replace('patch_embed3.', 'stages_2.downsample.')
                    elif 'patch_embed4.' in k:
                        new_key = k.replace('patch_embed4.', 'stages_3.downsample.')
                elif k.startswith('block'):
                    if 'block1.' in k:
                        new_key = k.replace('block1.', 'stages_0.blocks.')
                    elif 'block2.' in k:
                        new_key = k.replace('block2.', 'stages_1.blocks.')
                    elif 'block3.' in k:
                        new_key = k.replace('block3.', 'stages_2.blocks.')
                    elif 'block4.' in k:
                        new_key = k.replace('block4.', 'stages_3.blocks.')
                elif k.startswith('norm'):
                    if 'norm1.' in k:
                        new_key = k.replace('norm1.', 'stages_0.norm.')
                    elif 'norm2.' in k:
                        new_key = k.replace('norm2.', 'stages_1.norm.')
                    elif 'norm3.' in k:
                        new_key = k.replace('norm3.', 'stages_2.norm.')
                    elif 'norm4.' in k:
                        new_key = k.replace('norm4.', 'stages_3.norm.')
                if 'mlp.dwconv.dwconv.' in new_key:
                    new_key = new_key.replace('mlp.dwconv.dwconv.', 'mlp.dwconv.')
                
                new_state_dict[new_key] = v
            
            # --- 核心修正：将加载代码移出 for 循环 ---
            incompatible_keys = self.backbone.load_state_dict(new_state_dict, strict=False)
            
            print("\n\n--- Weight Loading Report ---")
            if incompatible_keys.missing_keys:
                print("Warning: Missing keys were not found in the checkpoint:", incompatible_keys.missing_keys)
            if incompatible_keys.unexpected_keys:
                print("Warning: Unexpected keys were found in the checkpoint:", incompatible_keys.unexpected_keys)
            print("\nBackbone weights (pvt_v2_b4, with NEW key fix) loaded successfully!")
            
        # --- 模型其他部分的定义 (保持在 __init__ 内部) ---
        self.reduce_s1 = Reduction(64, channel, RFB=False)
        self.reduce_s2 = Reduction(128, channel, RFB=False)
        self.reduce_s3 = Reduction(320, channel, RFB=False)
        self.reduce_s4 = Reduction(512, channel, RFB=False)

        self.reduce_e1 = Reduction(64, channel, RFB=False)
        self.reduce_e2 = Reduction(128, channel, RFB=False)
        self.reduce_e3 = Reduction(320, channel, RFB=False)
        self.reduce_e4 = Reduction(512, channel, RFB=False)

        self.learnable_edge_module = LearnableEdgeModule(in_channels=3, mid_channels=16, out_channels=1)
    
        self.Bconv1 = ConvBR(channel, channel,kernel_size=3, stride=1,padding=1)
        self.Bconv2 = ConvBR(channel, channel,kernel_size=3, stride=1,padding=1)
    
        self.iam1 = EIA(channel,channel,channel)
        self.hifm = HIFM(in_channels=channel, 
                         embed_dim=channel,
                         out_channels=channel, 
                         num_heads=4,
                         token_reduction_scale=4)
        self.iam3 = EIA(channel,channel,channel)
        self.sie1 = SEA(channel)
        self.sie2 = SEA(channel)
        self.sie3 = SEA(channel)

        self.rcab1 = RCAB(channel)
        self.rcab2 = RCAB(channel)
        self.rcab3 = RCAB(channel)
        self.rcab4 = RCAB(channel)
        self.rcab5 = RCAB(channel)
        self.rcab6 = RCAB(channel)

        self.guideflow_sh1 = Guide_flow(channel,channel)
        self.guideflow_sh2 = Guide_flow(channel,channel)
        self.guideflow_sh3 = Guide_flow(channel,channel)

        self.guideflow_eh1 = Guide_flow(channel,channel)
        self.guideflow_eh2 = Guide_flow(channel,channel)
        self.guideflow_eh3 = Guide_flow(channel,channel)

        self.pre_out1  = nn.Conv2d(channel, 1, 1)
        self.pre_out2  = nn.Conv2d(channel, 1, 1)
        self.pre_out3  = nn.Conv2d(channel, 1, 1)
        self.pre_out4  = nn.Conv2d(channel, 1, 1)
        self.pre_out5  = nn.Conv2d(channel, 1, 1)
        self.pre_out6  = nn.Conv2d(channel, 1, 1)
        
    def forward(self, x):
        shape = x.size()[2:]
        learned_edge_pred = self.learnable_edge_module(x)
    
        pvt_features = self.backbone(x)
    
        x1 = pvt_features[0]
        x2 = pvt_features[1]
        x3 = pvt_features[2]
        x4 = pvt_features[3]

        x1_t = self.reduce_s1(x1)
        x2_t = self.reduce_s2(x2)
        x3_t = self.reduce_s3(x3)
        x4_t = self.reduce_s4(x4)

        x1_e = self.reduce_e1(x1)
        x2_e = self.reduce_e2(x2)
        x3_e = self.reduce_e3(x3)
        x4_e = self.reduce_e4(x4)
    
        hr_s = self.guideflow_sh1(x4_t, x1_t)
        hr_s = self.rcab1(hr_s)
        hr_e = self.guideflow_eh1(x4_e, x1_e)
        hr_e = self.rcab4(hr_e)
        edge_1_raw = self.sie1(x4_t, x3_e, x4_e)

        learned_edge_resized_1 = F.interpolate(learned_edge_pred, size=edge_1_raw.shape[2:], mode='bilinear', align_corners=False)
        gate = torch.sigmoid(learned_edge_resized_1)
        edge_1 = edge_1_raw * (1 + gate)
        fuse_1 = self.iam1(x3_t, x4_t, edge_1)
    
        hr_s = self.guideflow_sh2(fuse_1, hr_s)
        hr_s = self.rcab2(hr_s)
        hr_e = self.guideflow_eh2(edge_1, hr_e)
        hr_e = self.rcab5(hr_e)
        edge_2_raw = self.sie2(fuse_1, x2_e, edge_1)
        learned_edge_resized_2 = F.interpolate(learned_edge_pred, size=edge_2_raw.shape[2:], mode='bilinear', align_corners=False)
        gate_2 = torch.sigmoid(learned_edge_resized_2) 
        edge_2 = edge_2_raw * (1 + gate_2)
        fuse_2, final_gate = self.hifm(x2_t, fuse_1, learned_edge_pred)
    
        hr_s = self.guideflow_sh3(fuse_2, hr_s)
        hr_s = self.rcab3(hr_s)
        hr_e = self.guideflow_eh3(edge_2, hr_e)
        hr_e = self.rcab6(hr_e)
        edge_3_raw = self.sie3(fuse_2, hr_e, edge_2)
        learned_edge_resized_3 = F.interpolate(learned_edge_pred, size=edge_3_raw.shape[2:], mode='bilinear', align_corners=False)
        gate_3 = torch.sigmoid(learned_edge_resized_3)
        edge_3 = edge_3_raw * (1 + gate_3)
        fuse_2_up = F.interpolate(fuse_2, size=hr_s.size()[2:], mode='bilinear', align_corners=True)
        fuse_3 = self.iam3(hr_s, fuse_2_up, edge_3)

        preds1   = F.interpolate(self.pre_out1(fuse_1), size=shape, mode='bilinear', align_corners=True) 
        preds2   = F.interpolate(self.pre_out2(fuse_2), size=shape, mode='bilinear', align_corners=True) 
        pred_f   = F.interpolate(self.pre_out3(fuse_3), size=shape, mode='bilinear', align_corners=True)

        prede1   = F.interpolate(self.pre_out4(edge_1), size=shape, mode='bilinear', align_corners=True) 
        prede2   = F.interpolate(self.pre_out5(edge_2), size=shape, mode='bilinear', align_corners=True) 
        prede3   = F.interpolate(self.pre_out6(edge_3), size=shape, mode='bilinear', align_corners=True)

        if self.training:
            return preds1, pred_f, preds2, prede1, prede2, prede3, learned_edge_pred, final_gate
        else:
            return preds1, pred_f, preds2, prede1, prede2, prede3, fuse_3, fuse_2, learned_edge_pred, final_gate
    
if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 384, 384)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
