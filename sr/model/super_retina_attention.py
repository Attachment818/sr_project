"""
SuperRetina模型 - 带注意力机制版本
这是一个独立的模型类，不影响原有的SuperRetina模型
可以安全地加载旧模型（会自动忽略注意力模块的参数）
"""
import random
import sys
import time

from model.pke_module import pke_learn
from model.attention_module import CBAM, ChannelAttention, SpatialAttention

from torch.nn import functional as F
import torch
import torch.nn as nn

from loss.dice_loss import DiceBCELoss, DiceLoss
from loss.triplet_loss import triplet_margin_loss_gor, triplet_margin_loss_gor_one, sos_reg

from common.common_util import remove_borders, sample_keypoint_desc, simple_nms, nms, \
    sample_descriptors
from common.train_util import get_gaussian_kernel, affine_images


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class SuperRetinaWithAttention(nn.Module):
    """
    SuperRetina模型 - 带注意力机制版本
    
    这个版本在原有SuperRetina基础上添加了注意力机制来增强血管特征。
    可以安全地加载旧的SuperRetina模型权重（会自动忽略注意力模块的参数）。
    
    参数:
        config: 配置字典
        device: 设备 ('cpu' 或 'cuda')
        n_class: 输出类别数，默认为1
        use_attention: 是否使用注意力机制，默认为True
        attention_type: 注意力类型，可选 'CBAM', 'Channel', 'Spatial'，默认为'CBAM'
    """
    def __init__(self, config=None, device='cpu', n_class=1, use_attention=True, attention_type='CBAM'):
        super().__init__()

        self.PKE_learn = True
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1, d2 = 64, 64, 128, 128, 256, 256, 256
        
        # Shared Encoder - 与原始SuperRetina完全相同
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)

        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)

        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)

        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # 注意力机制模块 - 仅在use_attention=True时创建
        # 修复：减少注意力模块数量，只在关键位置使用（避免过多注意力导致训练不稳定）
        # 只在编码器的高层（conv3, conv4）和检测器头的最后使用注意力
        if self.use_attention:
            if attention_type == 'CBAM':
                # CBAM结合通道和空间注意力，全面增强血管特征
                # 只在关键位置使用：编码器高层 + 检测器头最后
                self.attention3 = CBAM(c3, reduction=16, kernel_size=7, use_residual=True)
                self.attention4 = CBAM(c4, reduction=16, kernel_size=7, use_residual=True)
                # 只在检测器头的最后使用注意力
                self.attention_up1 = CBAM(c1, reduction=16, kernel_size=7, use_residual=True)
            elif attention_type == 'Channel':
                # 仅使用通道注意力
                self.attention3 = ChannelAttention(c3, reduction=16, use_residual=True)
                self.attention4 = ChannelAttention(c4, reduction=16, use_residual=True)
                self.attention_up1 = ChannelAttention(c1, reduction=16, use_residual=True)
            elif attention_type == 'Spatial':
                # 仅使用空间注意力
                self.attention3 = SpatialAttention(kernel_size=7, use_residual=True)
                self.attention4 = SpatialAttention(kernel_size=7, use_residual=True)
                self.attention_up1 = SpatialAttention(kernel_size=7, use_residual=True)

        # Descriptor Head - 与原始SuperRetina完全相同
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)
        self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)

        self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)

        # Detector Head - 与原始SuperRetina完全相同
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(c3 + c4, c3)
        self.dconv_up2 = double_conv(c2 + c3, c2)
        self.dconv_up1 = double_conv(c1 + c2, c1)

        self.conv_last = nn.Conv2d(c1, n_class, kernel_size=1)

        if config is not None:
            self.config = config

            self.nms_size = config['nms_size']
            self.nms_thresh = config['nms_thresh']
            self.scale = 8

            self.dice = DiceLoss()

            self.kernel = get_gaussian_kernel(kernlen=config['gaussian_kernel_size'],
                                              nsig=config['gaussian_sigma']).to(device)

        self.to(device)

    def network(self, x):
        # 编码器第一层
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)
        
        # 编码器第二层
        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)
        
        # 编码器第三层 - 在高层特征上应用注意力（修复：减少注意力数量）
        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        # 应用注意力机制增强血管特征
        if self.use_attention:
            conv3 = self.attention3(conv3)
        x = self.pool(conv3)
        
        # 编码器第四层 - 在最高层特征上应用注意力
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # 应用注意力机制
        if self.use_attention:
            x = self.attention4(x)

        # Descriptor Head - 不在描述子头前使用注意力（修复：避免影响描述子学习）
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        desc = self.trans_conv(desc)

        # Detector Head - 解码器
        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)
        cPa = self.dconv_up3(cPa)
        
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)
        cPa = self.dconv_up2(cPa)
        
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)
        cPa = self.dconv_up1(cPa)
        # 只在检测器头的最后应用注意力（修复：减少注意力数量）
        if self.use_attention:
            cPa = self.attention_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        return semi, desc

    def load_pretrained_weights(self, checkpoint_path, device='cpu', strict=False):
        """
        安全地加载预训练权重
        
        这个方法可以加载旧的SuperRetina模型权重，会自动忽略注意力模块的参数。
        
        参数:
            checkpoint_path: 检查点文件路径
            device: 设备
            strict: 是否严格匹配所有参数，默认为False（允许部分匹配）
        
        返回:
            加载的检查点字典
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint
        
        # 获取当前模型的状态字典
        model_dict = self.state_dict()
        
        # 过滤掉注意力模块的参数（如果旧模型没有这些参数）
        pretrained_dict = {}
        for k, v in state_dict.items():
            # 只加载基础网络参数，忽略注意力模块
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                else:
                    print(f"Warning: Shape mismatch for {k}: model {model_dict[k].shape} vs checkpoint {v.shape}")
            else:
                # 如果checkpoint中有但模型中没有的参数，跳过（可能是注意力模块）
                if 'attention' not in k:
                    print(f"Warning: Parameter {k} not found in model")
        
        # 加载匹配的参数
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        # 如果启用了注意力但权重中没有，打印提示
        if self.use_attention:
            attention_params = [k for k in model_dict.keys() if 'attention' in k]
            loaded_attention = [k for k in pretrained_dict.keys() if 'attention' in k]
            if len(loaded_attention) == 0 and len(attention_params) > 0:
                print(f"Info: Attention modules initialized randomly (not found in checkpoint). "
                      f"Total attention parameters: {len(attention_params)}")
        
        return checkpoint

    def descriptor_loss(self, detector_pred, label_point_positions, descriptor_pred,
                        affine_descriptor_pred, grid_inverse, affine_detector_pred=None):
        """
        calculate descriptor loss, construct triples on raw images and affine images
        :param detector_pred: output of detector network
        :param label_point_positions: initial label points
        :param descriptor_pred: output of descriptor network
        :param affine_descriptor_pred: output of descriptor network, with affine images as input
        :param grid_inverse: used for inverse affine transformation
        :return: descriptor loss (triplet loss)
        """

        if not self.PKE_learn:
            detector_pred[:] = 0  # only learn from the initial labels
        detector_pred[label_point_positions == 1] = 10
        descriptors, affine_descriptors, keypoints = \
            sample_descriptors(detector_pred, descriptor_pred, affine_descriptor_pred, grid_inverse,
                               nms_size=self.nms_size, nms_thresh=self.nms_thresh, scale=self.scale,
                               affine_detector_pred=affine_detector_pred)

        positive = []
        negatives_hard = []
        negatives_random = []
        anchor = []
        D = descriptor_pred.shape[1]
        for i in range(len(affine_descriptors)):
            if affine_descriptors[i].shape[1] == 0:
                continue
            descriptor = descriptors[i]  # (D, n)
            affine_descriptor = affine_descriptors[i]  # (D, n)

            n = affine_descriptors[i].shape[1]
            if n > 1000:  # avoid OOM
                return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

            descriptor = descriptor.view(D, -1, 1)  # (D, n, 1)
            affine_descriptor = affine_descriptor.view(D, 1, -1)  # (D, 1, n)
            ar = torch.arange(n)

            # random
            neg_index2 = []
            if n == 1:
                neg_index2.append(0)
            else:
                for j in range(n):
                    t = j
                    while t == j:
                        t = random.randint(0, n - 1)
                    neg_index2.append(t)
            neg_index2 = torch.tensor(neg_index2, dtype=torch.long).to(affine_descriptor)

            # hard - 优化显存：分批计算距离矩阵，避免创建大张量
            with torch.no_grad():
                # 分批计算距离，避免一次性创建 (n, n) 的大张量
                # descriptor: (D, n, 1), affine_descriptor: (D, 1, n)
                # 计算每个descriptor到所有affine_descriptor的距离
                batch_size_for_dist = min(100, n)  # 每次处理100个点
                neg_index1_list = []
                
                for start_idx in range(0, n, batch_size_for_dist):
                    end_idx = min(start_idx + batch_size_for_dist, n)
                    batch_size_actual = end_idx - start_idx
                    # 只计算当前batch的距离
                    desc_batch = descriptor[:, start_idx:end_idx, :]  # (D, batch_size, 1)
                    # 计算 (batch_size, n) 的距离矩阵
                    diff = desc_batch - affine_descriptor  # (D, batch_size, n)
                    dis_batch = torch.norm(diff, dim=0)  # (batch_size, n)
                    # 将对角线元素设为最大值+1（避免选择自己）
                    # dis_batch的第i行对应原始索引start_idx+i，对角线位置是(start_idx+i, start_idx+i)
                    local_indices = torch.arange(batch_size_actual, device=dis_batch.device)
                    orig_indices = torch.arange(start_idx, end_idx, device=dis_batch.device)
                    dis_batch[local_indices, orig_indices] = dis_batch.max() + 1
                    # 找到每个点的最近邻
                    neg_index1_list.append(dis_batch.argmin(dim=1))
                    # 释放中间变量
                    del diff, dis_batch
                
                neg_index1 = torch.cat(neg_index1_list)
                del neg_index1_list

            positive.append(affine_descriptor[:, 0, :].permute(1, 0))
            anchor.append(descriptor[:, :, 0].permute(1, 0))
            negatives_hard.append(affine_descriptor[:, 0, neg_index1.long(), ].permute(1, 0))
            negatives_random.append(affine_descriptor[:, 0, neg_index2.long(), ].permute(1, 0))
            
            # 释放不再需要的变量
            del descriptor, affine_descriptor, neg_index1, neg_index2

        if len(positive) == 0:
            return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

        positive = torch.cat(positive)
        anchor = torch.cat(anchor)
        negatives_hard = torch.cat(negatives_hard)
        negatives_random = torch.cat(negatives_random)

        positive = F.normalize(positive, dim=-1, p=2)
        anchor = F.normalize(anchor, dim=-1, p=2)
        negatives_hard = F.normalize(negatives_hard, dim=-1, p=2)
        negatives_random = F.normalize(negatives_random, dim=-1, p=2)

        loss = triplet_margin_loss_gor(anchor, positive, negatives_hard, negatives_random, margin=0.8)

        return loss, True

    def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
        """
        In interface phase, only need to input x
        :param x: retinal images
        :param label_point_positions: positions of keypoints on labels
        :param value_map: value maps, used to record history learned geo_points
        :param learn_index: index of input data with detector labels
        :param phase: distinguish dataset
        :return: if training, return loss, else return predictions
        """

        detector_pred, descriptor_pred = self.network(x)
        enhanced_label_pts = None
        enhanced_label = None

        if label_point_positions is not None:
            if self.PKE_learn:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = x.shape[0]
            else:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = loss_detector_num

            number_pts = 0  # number of learned keypoints
            value_map_update = None
            loss_detector = torch.tensor(0., requires_grad=True).to(x)
            loss_descriptor = torch.tensor(0., requires_grad=True).to(x)

            with torch.no_grad():
                affine_x, grid, grid_inverse = affine_images(x, used_for='detector')
                affine_detector_pred, affine_descriptor_pred = self.network(affine_x)
            loss_cal = self.dice
            if len(learn_index[0]) != 0:
                loss_detector, number_pts, value_map_update, enhanced_label_pts, enhanced_label = \
                    pke_learn(detector_pred[learn_index], descriptor_pred[learn_index],
                              grid_inverse[learn_index], affine_detector_pred[learn_index],
                              affine_descriptor_pred[learn_index], self.kernel, loss_cal,
                              label_point_positions[learn_index], value_map[learn_index],
                              self.config, self.PKE_learn)

            #  For showing PKE process
            if enhanced_label_pts is not None:
                enhanced_label_pts_tmp = label_point_positions.clone()
                enhanced_label_pts_tmp[learn_index] = enhanced_label_pts
                enhanced_label_pts = enhanced_label_pts_tmp
            if enhanced_label is not None:
                enhanced_label_tmp = label_point_positions.clone()
                enhanced_label_tmp[learn_index] = enhanced_label
                enhanced_label = enhanced_label_tmp

            detector_pred_copy = detector_pred.clone().detach()
            affine_x_for_desc, grid_for_desc, grid_inverse_for_desc = affine_images(x, used_for='descriptor')
            _, affine_descriptor_pred_for_desc = self.network(affine_x_for_desc)
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(detector_pred_copy, label_point_positions,
                                                                          descriptor_pred,
                                                                          affine_descriptor_pred_for_desc,
                                                                          grid_inverse_for_desc)

            if self.PKE_learn and len(learn_index[0]) != 0:
                value_map[learn_index] = value_map_update
            loss = loss_detector + loss_descriptor

            return loss, number_pts, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

        return detector_pred, descriptor_pred
