# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint

from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)
from mmedit.models.common import PixelShufflePack, flow_warp
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class OnRefVSR(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output.

    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_pretrained=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
            self.feat_extract_hr1 = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['forward_1', 'forward_2']
        # modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            3 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False
        self.process_hr_feature1 = DownsampleModel(mid_channels)
        self.LNT = LNT_block_PS(6, 96) # 1*1*6*16
        self.LFTA = nn.Sequential(
            nn.Conv2d(64 * 2, 64, 1, 1, bias=True), nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 1, 1, bias=True), nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 1, 1, bias=True), nn.Sigmoid())
        self.LPE = nn.Sequential(
        nn.Conv2d(64*2, 64, 1, 1, bias=True), nn.LeakyReLU(0.1, True),
        nn.Conv2d(64, 64, 1, 1, bias=True), nn.LeakyReLU(0.1, True),
        nn.Conv2d(64, 64*2, 1, 1, bias=True), nn.Sigmoid())
        self.AttenFusion = nn.Sequential(
        nn.Conv2d(64*2, 64, 1, 1, bias=True), nn.LeakyReLU(0.1, True), nn.Sigmoid())
        self.fuse_para_lnt = nn.Sequential(
            nn.Conv2d(6 * 2, 6, 1, 1, bias=True), nn.LeakyReLU(0.1, True),
            nn.Conv2d(6, 6, 1, 1, bias=True), nn.LeakyReLU(0.1, True),
            nn.Conv2d(6, 6, 1, 1, bias=True), nn.Sigmoid())
        self.RLPE = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels* 2, 1),
            nn.Sigmoid()
        )
        self.conv_fusion= nn.Conv2d(6, 3, 3, 1, 1)


    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        # if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
        #     flows_forward = None
        # else:
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward

    def propagate(self, feats, flows, module_name,para):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                w = self.RLPE(torch.cat([cond_n1, feat_current], dim=1))
                k, b = torch.chunk(w, 2, dim=1)
                cond_n1 = cond_n1 * k + b
               
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2,para)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def gen_warp_para(self, para, flows):
        n, t, c, h, w = flows.size()
        para_list = []
        for i in range(1, t+1):
            para = flow_warp(para, flows[:, i-1, :, :, :].permute(0, 2, 3, 1))
            para_list.append(para)
        return torch.stack(para_list, dim=1)

    def upsample(self, lqs, feats, gt0,feats_gt0_hr,flows):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
           
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))

            # extract features and propagate
            if i != 0:
                hr = self.LFTA(torch.cat([hr, feature_last], dim=1)) * hr + hr
            else:
                hr = self.LFTA(torch.cat([hr, feats_gt0_hr], dim=1)) * hr + hr
            feature_last = hr  

            hr = self.conv_last(hr)

            if i==0:
                map = self.LNT(gt0,hr,1)
                n, t, c, h, w = flows.size()
                flows = F.interpolate(flows.view(-1,c,h,w), scale_factor=4, mode='bilinear', align_corners=False).view(n,t,c,h*4,w*4)
                map_warp = self.gen_warp_para(map, flows)
                map_a,map_x0,map_k,map_q,map_c,map_e = map[:,:1],map[:,1:2],map[:,2:3],map[:,3:4],map[:,4:5],map[:,5:6]
                hr = gt0

            else:
                map_a,map_x0,map_k,map_q,map_c,map_e = map_warp[:, i-1, :, :, :][:,:1],map_warp[:, i-1, :, :, :][:,1:2],map_warp[:, i-1, :, :, :][:,2:3],map_warp[:, i-1, :, :, :][:,3:4],map_warp[:, i-1, :, :, :][:,4:5],map_warp[:, i-1, :, :, :][:,5:6]
                hr = map_a / (1 + map_q*torch.exp(map_k * (hr - map_x0)))  + map_c + map_e * hr

            hr = self.conv_fusion(torch.cat([self.img_upsample(lqs[:, i, :, :, :]),hr],dim=1))

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs,gt):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)

        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w)).view(n,t,-1,h,w)
            feats_gt0_hr = self.feat_extract_hr1(gt[:,0,:,:,:])
            feats_gt0 = self.process_hr_feature1(feats_gt0_hr)#[1, 64, 64, 64]
            para = self.LPE(torch.cat([feats_gt0,feats_[:,0,:,:,:]],dim=1))

           
        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward = self.compute_flow(lqs_downsample)
       
        para_warp = self.gen_warp_para(para, flows_forward)
        feats['spatial'] =[]
        for i in range(0, t):
            if i == 0:
                feats['spatial'].append(feats_gt0)
            else:
                new_feat = feats_[:,i,:,:,:]+feats_[:,i,:,:,:]*self.AttenFusion(torch.cat([feats_[:,i,:,:,:]*para_warp[:, i-1, :64, :, :]+para_warp[:, i-1, 64:, :, :],feats_[:,i,:,:,:]],dim=1))
                feats['spatial'].append(new_feat)

        for iter_ in [1, 2]:
            for direction in ['forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []
                flows = flows_forward
                feats = self.propagate(feats, flows, module,para)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats, gt[:,0,:,:,:],feats_gt0_hr,flows_forward)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(5 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2,para):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2,para], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
   
class DownsampleModel(nn.Module):
    def __init__(self, in_channels):
        super(DownsampleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)  # [b, 64, 128, 128]
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)  # [b, 64, 64, 64]
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x
   

class LNT_block_PS(nn.Module):
     def __init__(self, in_planes, c=128):
         super(LNT_block_PS, self).__init__()
         self.conv0 = nn.Sequential(
             nn.Conv2d(in_planes, c//2, kernel_size=3, stride=2, padding=1),
             nn.Conv2d(c//2, c, kernel_size=3, stride=2, padding=1),
             )
         self.convblock1 = nn.Sequential(
             nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
             nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
         )
         self.convblock2 = nn.Sequential(
             nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
             nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
         )

         self.ps = nn.PixelShuffle(4)


     def forward(self, x_,y_, scale= 1):
         if scale != 1:
             x = F.interpolate(x_, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
         x = torch.cat((x_,y_),dim=1)
         x = self.conv0(x)
         x = self.convblock1(x) + x 
         x = self.convblock2(x)
         tmp = self.ps(x)
         

         return tmp
     