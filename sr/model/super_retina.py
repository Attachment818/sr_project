import copy
import math
import random
import sys
import time

from model.pke_module import pke_learn, mapping_points, geometric_filter, content_filter

from torch.nn import functional as F
import torch
import torch.nn as nn

from loss.dice_loss import DiceBCELoss, DiceLoss, MaskedDiceLoss
from loss.triplet_loss import triplet_margin_loss_gor, triplet_margin_loss_gor_one, sos_reg
from loss.perceptual_loss import PerceptualLoss

from common.common_util import remove_borders, sample_keypoint_desc, simple_nms, nms, \
    sample_descriptors
from common.train_util import get_gaussian_kernel, affine_images
from common.vessel_mask_util import compute_vessel_mask_batch
from common.pke_diagnostics import summarize_pke_stages

# 导入自注意力模块（如果存在）
try:
    from model.attention_module import SelfAttention
    HAS_SELF_ATTENTION = True
except ImportError:
    HAS_SELF_ATTENTION = False


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


def chunked_hard_negative_indices(descriptor, affine_descriptor, keypoints=None,
                                  min_negative_distance=0.0, chunk_size=256):
    """Find the same nearest non-corresponding descriptors with bounded memory.

    Args:
        descriptor: [D, N] anchor descriptors.
        affine_descriptor: [D, N] candidate descriptors.
        keypoints: optional [N, 2] coordinates for spatial exclusion.
    """
    if descriptor.ndim != 2 or affine_descriptor.ndim != 2:
        raise ValueError('descriptor inputs must have shape [D, N]')
    if descriptor.shape != affine_descriptor.shape:
        raise ValueError('anchor and affine descriptor shapes must match')
    if chunk_size <= 0:
        raise ValueError('descriptor hard-negative chunk size must be positive')
    n = descriptor.shape[1]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=descriptor.device)
    if min_negative_distance > 0:
        if keypoints is None or keypoints.shape != (n, 2):
            raise RuntimeError(
                'Descriptor hard-negative spatial coordinates must align with '
                'the sampled descriptor matrix'
            )

    selected = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        distances = torch.norm(
            descriptor[:, start:end, None] - affine_descriptor[:, None, :], dim=0
        )
        local_rows = torch.arange(end - start, device=distances.device)
        global_rows = torch.arange(start, end, device=distances.device)
        distances[local_rows, global_rows] = distances.max() + 1
        if min_negative_distance > 0:
            spatial = torch.cdist(
                keypoints[start:end].float(), keypoints.float()
            )
            distances[spatial < min_negative_distance] = distances.max() + 1
        selected.append(distances.argmin(dim=1))
    return torch.cat(selected, dim=0)


class SuperRetina(nn.Module):
    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__()

        self.PKE_learn = True
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1, d2 = 64, 64, 128, 128, 256, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)

        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)

        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)

        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)
        self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)

        self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)

        # Detector Head (U-Net style decoder).
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(c3 + c4, c3)
        self.dconv_up2 = double_conv(c2 + c3, c2)
        self.dconv_up1 = double_conv(c1 + c2, c1)

        self.conv_last = nn.Conv2d(c1, n_class, kernel_size=1)

        if config is not None:
            self.config = config
            self._capture_descriptor_supervision_audit = bool(
                config.get('log_descriptor_supervision_stats', False)
            )

            self.nms_size = config['nms_size']
            self.nms_thresh = config['nms_thresh']
            self.scale = 8

            self.dice = DiceLoss()

            self.kernel = get_gaussian_kernel(kernlen=config['gaussian_kernel_size'],
                                              nsig=config['gaussian_sigma']).to(device)

        self.to(device)

    def network(self, x):
        """
        原始 SuperRetina 的单尺度解码结构（保持不变，作为 baseline）。
        """
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)

        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)

        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        desc = self.trans_conv(desc)

        # Detector Head（原有上采样+拼接结构）
        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)

        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        return semi, desc

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

        # sample keypoints on initial labels
        # label_descriptors, label_affine_descriptors, label_keypoints = \
        #     sample_descriptors(label_point_positions, descriptor_pred, affine_descriptor_pred, grid_inverse,
        #                        nms_size=self.nms_size, nms_thresh=self.nms_thresh, scale=self.scale)
        #
        # for s, kps in enumerate(label_keypoints):
        #     label_mask = torch.zeros(detector_pred[s].shape).to(detector_pred)
        #     label_mask[0, kps[:, 1].long(), kps[:, 0].long()] = 1
        #     label_mask = F.conv2d(label_mask.unsqueeze(0), self.mask_kernel, stride=1,
        #                           padding=(self.mask_kernel.shape[-1] - 1) // 2)
        #     detector_pred[s][label_mask[0] > 1e-5] = 0
        if not self.PKE_learn:
            detector_pred[:] = 0  # only learn from the initial labels
        detector_pred[label_point_positions == 1] = 10
        descriptors, affine_descriptors, keypoints = \
            sample_descriptors(detector_pred, descriptor_pred, affine_descriptor_pred, grid_inverse,
                               nms_size=self.nms_size, nms_thresh=self.nms_thresh, scale=self.scale,
                               affine_detector_pred=affine_detector_pred,
                               return_valid_keypoints=True)

        # Read-only audits can inspect why descriptor supervision is accepted or
        # skipped. The flag is absent during normal training, so legacy behavior
        # and return values remain unchanged.
        if getattr(self, '_capture_descriptor_supervision_audit', False):
            sample_counts = [int(item.shape[1]) for item in affine_descriptors]
            hard_negative_mode = self.config.get(
                'descriptor_hard_negative_mode', 'legacy'
            )
            over_limit_indices = [
                index for index, count in enumerate(sample_counts) if count > 1000
            ]
            nonempty_indices = [
                index for index, count in enumerate(sample_counts) if count > 0
            ]
            if over_limit_indices and hard_negative_mode == 'legacy':
                exit_reason = 'over_limit_batch_abort'
                participating_indices = []
            elif nonempty_indices:
                exit_reason = 'trained'
                participating_indices = nonempty_indices
            else:
                exit_reason = 'all_images_empty'
                participating_indices = []
            self._descriptor_supervision_audit = {
                'sample_counts': sample_counts,
                'over_limit_indices': over_limit_indices,
                'nonempty_indices': nonempty_indices,
                'participating_indices': participating_indices,
                'exit_reason': exit_reason,
                'sample_limit': 1000,
                'hard_negative_mode': hard_negative_mode,
            }

        # descriptors_tmp = []
        # affine_descriptor_tmp = []
        # for i in range(len(descriptors)):
        #     descriptors_tmp.append(torch.cat((descriptors[i], label_descriptors[i]), -1))
        #     affine_descriptor_tmp.append(torch.cat((affine_descriptors[i], label_affine_descriptors[i]), -1))
        # descriptors = descriptors_tmp
        # affine_descriptors = affine_descriptor_tmp

        positive = []
        negatives_hard = []
        negatives_random = []
        anchor = []
        D = descriptor_pred.shape[1]
        for i in range(len(affine_descriptors)):
            if affine_descriptors[i].shape[1] == 0:
                continue
            descriptor = descriptors[i]
            affine_descriptor = affine_descriptors[i]

            n = affine_descriptors[i].shape[1]
            hard_negative_mode = self.config.get(
                'descriptor_hard_negative_mode', 'legacy'
            )
            if hard_negative_mode not in {'legacy', 'chunked'}:
                raise ValueError(
                    f'Unknown descriptor_hard_negative_mode: {hard_negative_mode}'
                )
            if hard_negative_mode == 'legacy' and n > 1000:  # historical OOM guard
                return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

            descriptor = descriptor.view(D, -1, 1)
            affine_descriptor = affine_descriptor.view(D, 1, -1)
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

            # hard
            with torch.no_grad():
                min_negative_distance = float(self.config.get('descriptor_hard_negative_min_distance', 0.0))
                if hard_negative_mode == 'chunked':
                    neg_index1 = chunked_hard_negative_indices(
                        descriptors[i], affine_descriptors[i], keypoints=keypoints[i],
                        min_negative_distance=min_negative_distance,
                        chunk_size=int(self.config.get(
                            'descriptor_hard_negative_chunk_size', 256
                        )),
                    )
                else:
                    dis = torch.norm(descriptor - affine_descriptor, dim=0)
                    dis[ar, ar] = dis.max() + 1
                    if min_negative_distance > 0:
                        spatial = torch.cdist(keypoints[i].float(), keypoints[i].float())
                        if spatial.shape != dis.shape:
                            raise RuntimeError(
                                'Descriptor hard-negative spatial coordinates must align with '
                                'the sampled descriptor matrix'
                            )
                        dis[spatial < min_negative_distance] = dis.max() + 1
                    neg_index1 = dis.argmin(axis=1)

            positive.append(affine_descriptor[:, 0, :].permute(1, 0))
            anchor.append(descriptor[:, :, 0].permute(1, 0))
            negatives_hard.append(affine_descriptor[:, 0, neg_index1.long(), ].permute(1, 0))
            negatives_random.append(affine_descriptor[:, 0, neg_index2.long(), ].permute(1, 0))

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

        # can also add sos reg term .
        # reg_term = sos_reg(anchor, positive, KNN=True, k=1, eps=1e-8)
        # if not torch.isnan(reg_term) and reg_term > 0:
        #     loss = loss + 0.1 * reg_term

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
            # if value_map_update is not None:
            #     # optimize descriptors of recorded points
            #     detector_pred_copy[learn_index][value_map_update >=
            #                                     self.config['VALUE MAP'].getfloat('value_increase_point')] = 1
            #
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


class SuperRetinaFPN(SuperRetina):
    """
    带简单 FPN 多尺度解码头的 SuperRetina 变体。

    说明：
    - 完全复用原始编码器和描述子头，只替换检测头为 FPN 风格；
    - 不改动原始 SuperRetina 的实现，方便随时切回 baseline。
    """

    def __init__(self, config=None, device='cpu', n_class=1):
        # 先调用父类构造函数（包含 encoder + descriptor + 原始 decoder，并执行 self.to(device)）
        super().__init__(config=config, device=device, n_class=n_class)

        # 这里显式写出通道数，保持与父类一致
        c1, c2, c3, c4 = 64, 64, 128, 128
        # FPN 统一的通道数（各尺度都映射到这个维度，避免通道不匹配）
        c_fpn = 128

        # FPN 的横向 1x1 conv，将不同尺度特征映射到统一通道数 c_fpn
        self.lateral4 = nn.Conv2d(c4, c_fpn, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3, c_fpn, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2, c_fpn, kernel_size=1)
        self.lateral1 = nn.Conv2d(c1, c_fpn, kernel_size=1)

        # FPN 输出头：在最高分辨率的 FPN 特征上预测 keypoint heatmap
        self.fpn_out_conv = nn.Conv2d(c_fpn, n_class, kernel_size=1)

        # 将新增的 FPN 层也移动到指定 device，避免出现 CPU / CUDA 混用错误
        self.to(device)

    def network(self, x):
        """
        与原始网络接口完全一致：返回 (detector_pred, descriptor_pred)，
        区别在于 detector_pred 由 FPN 融合多尺度特征得到。
        """
        # 编码器（与父类完全相同）
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))        # H x W
        x = self.pool(conv1)                     # H/2 x W/2

        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))        # H/2 x W/2
        x = self.pool(conv2)                     # H/4 x W/4

        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))        # H/4 x W/4
        x = self.pool(conv3)                     # H/8 x W/8

        x = self.relu(self.conv4a(x))
        conv4 = self.relu(self.conv4b(x))        # H/8 x W/8

        # ===== Descriptor Head（保持不变，用最深层特征） =====
        cDa = self.relu(self.convDa(conv4))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)

        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        desc = self.trans_conv(desc)

        # ===== FPN Top-down 多尺度融合用于 Detector Head =====
        # 自顶向下：P4 -> P3 -> P2 -> P1（最终分辨率与输入一致）
        p4 = self.lateral4(conv4)                        # H/8, C_fpn
        p3 = self.lateral3(conv3) + self.upsample(p4)    # H/4, C_fpn
        p2 = self.lateral2(conv2) + self.upsample(p3)    # H/2, C_fpn
        p1 = self.lateral1(conv1) + self.upsample(p2)    # H,   C_fpn

        # 直接在最高分辨率的 FPN 特征上预测 keypoint heatmap
        semi = self.fpn_out_conv(p1)
        semi = torch.sigmoid(semi)

        return semi, desc


class SuperRetinaWithSelfAttention(SuperRetina):
    """
    带自注意力机制的 SuperRetina 模型
    
    在编码器的高层特征（conv3, conv4）上添加自注意力机制，用于：
    - 捕获长距离依赖（如血管的全局结构）
    - 学习位置间关系（如血管分叉点、交叉点）
    
    特点：
    - 完全向后兼容：可以加载原始 SuperRetina 的权重
    - 自注意力模块随机初始化
    - 只在关键位置使用（避免计算量过大）
    """
    def __init__(self, config=None, device='cpu', n_class=1, use_self_attention=True, attention_reduction=8):
        """
        Args:
            config: 配置字典
            device: 设备
            n_class: 输出类别数
            use_self_attention: 是否使用自注意力
            attention_reduction: 自注意力的降维比例（用于减少计算量）
        """
        # 先初始化父类（不调用super().__init__，因为我们要自定义）
        nn.Module.__init__(self)
        
        if not HAS_SELF_ATTENTION:
            raise ImportError("SelfAttention module not found. Please ensure attention_module.py contains SelfAttention class.")
        
        self.PKE_learn = True
        self.use_self_attention = use_self_attention
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

        # 自注意力模块 - 只在编码器最高层使用（减少显存占用）
        # 只在conv4之后使用，因为：
        # 1. conv4分辨率最低（H/8），显存占用最小
        # 2. 高层特征语义更丰富，自注意力效果更好
        # 3. conv3分辨率较高（H/4），计算注意力矩阵会占用过多显存
        if self.use_self_attention:
            # 只在conv4使用自注意力，避免显存溢出
            self.self_attention4 = SelfAttention(c4, reduction=attention_reduction, use_residual=True)

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
        """
        带自注意力的网络前向传播
        在编码器高层（conv3, conv4）应用自注意力
        """
        # 编码器第一层
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)
        
        # 编码器第二层
        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)
        
        # 编码器第三层 - 不使用自注意力（分辨率太高，显存占用大）
        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)
        
        # 编码器第四层 - 应用自注意力（分辨率最低，显存占用可接受）
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        if self.use_self_attention:
            x = self.self_attention4(x)

        # Descriptor Head - 与原始SuperRetina完全相同
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        desc = self.trans_conv(desc)

        # Detector Head - 与原始SuperRetina完全相同
        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)

        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        return semi, desc
    
    def load_pretrained_weights(self, checkpoint_path, device='cpu', strict=False):
        """
        安全加载预训练权重
        自动忽略自注意力模块的参数（如果旧模型没有）
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取模型状态字典
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        else:
            pretrained_dict = checkpoint
        
        # 获取当前模型状态字典
        model_dict = self.state_dict()
        
        # 过滤掉不匹配的键（自注意力模块）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新模型字典
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        # 统计信息
        total_params = len(model_dict)
        matched_params = len(pretrained_dict)
        
        # 检查自注意力参数：当前模型中有，但预训练权重中没有的
        model_attention_params = [k for k in model_dict.keys() if 'self_attention' in k]
        pretrained_attention_params = [k for k in pretrained_dict.keys() if 'self_attention' in k]
        missing_attention_params = [k for k in model_attention_params if k not in pretrained_attention_params]
        
        print(f"Loaded pretrained weights from {checkpoint_path}")
        print(f"  Matched: {matched_params}/{total_params} parameters")
        if len(missing_attention_params) > 0:
            print(f"  Self-attention modules ({len(missing_attention_params)} params) are randomly initialized "
                  f"(not found in checkpoint)")
        elif len(model_attention_params) > 0:
            print(f"  Self-attention modules ({len(model_attention_params)} params) loaded from checkpoint")
        
        return checkpoint


class SuperRetinaWithMultiScaleDescriptor(SuperRetina):
    """
    带多尺度描述子融合的 SuperRetina 模型
    
    改进点：
    - 描述子头融合conv2、conv3、conv4的多尺度特征
    - 结合细节（conv2）和语义（conv4）信息
    - 对尺度变化更鲁棒，提升匹配准确率
    
    特点：
    - 完全向后兼容：可以加载原始 SuperRetina 的权重
    - 多尺度融合模块随机初始化
    - 只修改描述子头，检测器头保持不变
    """
    def __init__(self, config=None, device='cpu', n_class=1, use_multi_scale_desc=True):
        """
        Args:
            config: 配置字典
            device: 设备
            n_class: 输出类别数
            use_multi_scale_desc: 是否使用多尺度描述子融合
        """
        # 先初始化父类（不调用super().__init__，因为我们要自定义）
        nn.Module.__init__(self)
        
        self.PKE_learn = True
        self.use_multi_scale_desc = use_multi_scale_desc
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

        # Descriptor Head - 多尺度融合版本
        if self.use_multi_scale_desc:
            # 多尺度特征融合
            # conv2: [B, 64, H/2, W/2] -> 下采样到 H/8
            # conv3: [B, 128, H/4, W/4] -> 下采样到 H/8
            # conv4: [B, 128, H/8, W/8] -> 已经是 H/8
            
            # 将conv2和conv3下采样到H/8分辨率
            self.desc_downsample2 = nn.Sequential(
                nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1),  # H/2 -> H/4
                nn.ReLU(inplace=True),
                nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1),  # H/4 -> H/8
                nn.ReLU(inplace=True)
            )
            
            self.desc_downsample3 = nn.Sequential(
                nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1),  # H/4 -> H/8
                nn.ReLU(inplace=True)
            )
            
            # 融合多尺度特征：conv2(64) + conv3(128) + conv4(128) = 320通道
            desc_fusion_channels = c2 + c3 + c4  # 64 + 128 + 128 = 320
            self.desc_fusion = nn.Sequential(
                nn.Conv2d(desc_fusion_channels, c5, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            # 不使用多尺度融合，与原始SuperRetina相同
            self.desc_downsample2 = None
            self.desc_downsample3 = None
            self.desc_fusion = None
        
        # 后续的描述子处理（与原始SuperRetina相同，无论是否使用多尺度融合）
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
        """
        带多尺度描述子融合的网络前向传播
        描述子头融合conv2、conv3、conv4的特征
        """
        # 编码器（与原始SuperRetina完全相同）
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)
        
        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)
        
        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)
        
        x = self.relu(self.conv4a(x))
        conv4 = self.relu(self.conv4b(x))

        # Descriptor Head - 多尺度融合版本
        if self.use_multi_scale_desc:
            # 将conv2和conv3下采样到H/8分辨率（与conv4相同）
            desc2 = self.desc_downsample2(conv2)  # [B, 64, H/8, W/8]
            desc3 = self.desc_downsample3(conv3)  # [B, 128, H/8, W/8]
            desc4 = conv4  # [B, 128, H/8, W/8]
            
            # 融合多尺度特征
            desc_fused = torch.cat([desc2, desc3, desc4], dim=1)  # [B, 320, H/8, W/8]
            desc_fused = self.desc_fusion(desc_fused)  # [B, 256, H/8, W/8]
            
            # 后续处理（与原始SuperRetina相同）
            cDb = self.relu(self.convDb(desc_fused))
            desc = self.convDc(cDb)
        else:
            # 不使用多尺度融合，与原始SuperRetina相同
            cDa = self.relu(self.convDa(conv4))
            cDb = self.relu(self.convDb(cDa))
            desc = self.convDc(cDb)

        # L2归一化
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # 上采样回H/8
        desc = self.trans_conv(desc)

        # Detector Head - 与原始SuperRetina完全相同
        cPa = self.upsample(conv4)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)

        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        return semi, desc
    
    def load_pretrained_weights(self, checkpoint_path, device='cpu', strict=False):
        """
        安全加载预训练权重
        自动忽略多尺度描述子模块的参数（如果旧模型没有）
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取模型状态字典
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        else:
            pretrained_dict = checkpoint
        
        # 获取当前模型状态字典
        model_dict = self.state_dict()
        
        # 过滤掉不匹配的键（多尺度描述子模块）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新模型字典
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        # 统计信息
        total_params = len(model_dict)
        matched_params = len(pretrained_dict)
        
        # 检查多尺度描述子参数
        model_multiscale_params = [k for k in model_dict.keys() 
                                  if 'desc_downsample' in k or 'desc_fusion' in k]
        pretrained_multiscale_params = [k for k in pretrained_dict.keys() 
                                       if 'desc_downsample' in k or 'desc_fusion' in k]
        missing_multiscale_params = [k for k in model_multiscale_params 
                                     if k not in pretrained_multiscale_params]
        
        print(f"Loaded pretrained weights from {checkpoint_path}")
        print(f"  Matched: {matched_params}/{total_params} parameters")
        if len(missing_multiscale_params) > 0:
            print(f"  Multi-scale descriptor modules ({len(missing_multiscale_params)} params) are randomly initialized "
                  f"(not found in checkpoint)")
        elif len(model_multiscale_params) > 0:
            print(f"  Multi-scale descriptor modules ({len(model_multiscale_params)} params) loaded from checkpoint")
        
        return checkpoint


class ASPPModule(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) 模块
    使用多个不同膨胀率的空洞卷积，捕获多尺度感受野
    
    特点：
    - 不增加分辨率，显存友好
    - 多尺度感受野
    - 对血管粗细变化敏感
    """
    def __init__(self, in_channels, out_channels, rates=[1, 2, 4, 8]):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            rates: 空洞卷积的膨胀率列表，默认[1, 2, 4, 8]
        """
        super(ASPPModule, self).__init__()
        self.rates = rates
        
        # 不同膨胀率的空洞卷积
        self.aspp_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, 
                     padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, 
                     padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, 
                     padding=rates[3], dilation=rates[3], bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 全局平均池化分支
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 融合所有分支
        # 5个分支拼接：每个分支输出 out_channels//4 通道，共 5*(out_channels//4) 通道
        aspp_fusion_in_channels = 5 * (out_channels // 4)
        self.aspp_fusion = nn.Sequential(
            nn.Conv2d(aspp_fusion_in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            out: 增强后的特征 [B, C', H, W]
        """
        B, C, H, W = x.size()
        
        # 不同膨胀率的卷积
        aspp1 = self.aspp_conv1(x)  # rate=1, 标准卷积
        aspp2 = self.aspp_conv2(x)  # rate=2
        aspp3 = self.aspp_conv3(x)  # rate=4
        aspp4 = self.aspp_conv4(x)  # rate=8
        
        # 全局平均池化
        aspp_pool = self.aspp_pool(x)  # [B, C//4, 1, 1]
        aspp_pool = F.interpolate(aspp_pool, size=(H, W), mode='bilinear', align_corners=True)  # 上采样到原尺寸
        
        # 拼接所有分支
        aspp_out = torch.cat([aspp1, aspp2, aspp3, aspp4, aspp_pool], dim=1)  # [B, C, H, W]
        
        # 融合
        out = self.aspp_fusion(aspp_out)
        
        return out


class SuperRetinaWithASPP(SuperRetina):
    """
    带ASPP模块的 SuperRetina 模型
    
    改进点：
    - 在描述子头输入前添加ASPP模块
    - 使用多个不同膨胀率的空洞卷积，捕获多尺度感受野
    - 对血管粗细变化更敏感，提升描述子质量
    
    特点：
    - 完全向后兼容：可以加载原始 SuperRetina 的权重
    - ASPP模块随机初始化
    - 只修改描述子头输入，其他部分保持不变
    """
    def __init__(self, config=None, device='cpu', n_class=1, use_aspp=True, aspp_rates=[1, 2, 4, 8]):
        """
        Args:
            config: 配置字典
            device: 设备
            n_class: 输出类别数
            use_aspp: 是否使用ASPP模块
            aspp_rates: ASPP的膨胀率列表，默认[1, 2, 4, 8]
        """
        # 先初始化父类（不调用super().__init__，因为我们要自定义）
        nn.Module.__init__(self)
        
        self.PKE_learn = True
        self.use_aspp = use_aspp
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

        # ASPP模块 - 在描述子头输入前使用
        if self.use_aspp:
            # ASPP输入是conv4的输出（128通道），输出也是128通道（保持与原始模型兼容）
            self.aspp = ASPPModule(in_channels=c4, out_channels=c4, rates=aspp_rates)
        else:
            self.aspp = None

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
        """
        带ASPP的网络前向传播
        在描述子头输入前应用ASPP模块
        """
        # 编码器（与原始SuperRetina完全相同）
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)
        
        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)
        
        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)
        
        x = self.relu(self.conv4a(x))
        conv4 = self.relu(self.conv4b(x))

        # 应用ASPP模块（如果启用）
        if self.use_aspp:
            conv4_enhanced = self.aspp(conv4)  # 增强conv4特征
        else:
            conv4_enhanced = conv4

        # Descriptor Head - 使用增强后的特征
        cDa = self.relu(self.convDa(conv4_enhanced))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)

        # L2归一化
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # 上采样回H/8
        desc = self.trans_conv(desc)

        # Detector Head - 使用原始conv4（保持检测器头不变）
        cPa = self.upsample(conv4)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)

        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        return semi, desc
    
    def load_pretrained_weights(self, checkpoint_path, device='cpu', strict=False):
        """
        安全加载预训练权重
        自动忽略ASPP模块的参数（如果旧模型没有）
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取模型状态字典
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        else:
            pretrained_dict = checkpoint
        
        # 获取当前模型状态字典
        model_dict = self.state_dict()
        
        # 过滤掉不匹配的键（ASPP模块）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新模型字典
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        # 统计信息
        total_params = len(model_dict)
        matched_params = len(pretrained_dict)
        
        # 检查ASPP参数
        model_aspp_params = [k for k in model_dict.keys() if 'aspp' in k.lower()]
        pretrained_aspp_params = [k for k in pretrained_dict.keys() if 'aspp' in k.lower()]
        missing_aspp_params = [k for k in model_aspp_params if k not in pretrained_aspp_params]
        
        print(f"Loaded pretrained weights from {checkpoint_path}")
        print(f"  Matched: {matched_params}/{total_params} parameters")
        if len(missing_aspp_params) > 0:
            print(f"  ASPP modules ({len(missing_aspp_params)} params) are randomly initialized "
                  f"(not found in checkpoint)")
        elif len(model_aspp_params) > 0:
            print(f"  ASPP modules ({len(model_aspp_params)} params) loaded from checkpoint")
        
        return checkpoint


class SuperRetinaWithoutPKE(SuperRetina):
    """
    不使用PKE模块的 SuperRetina 模型
    
    改进点：
    - 完全去掉PKE学习机制
    - 只使用初始标注的关键点进行训练
    - 简化训练流程，减少计算开销
    
    特点：
    - 完全向后兼容：可以加载原始 SuperRetina 的权重
    - 训练更简单：不需要value_map和affine变换用于PKE学习
    - 计算更快：减少了PKE相关的计算
    - 只使用初始标签：直接使用标注的关键点，不学习额外关键点
    """
    def __init__(self, config=None, device='cpu', n_class=1):
        """
        Args:
            config: 配置字典
            device: 设备
            n_class: 输出类别数
        """
        # 先初始化父类（不调用super().__init__，因为我们要自定义）
        nn.Module.__init__(self)
        
        self.PKE_learn = False  # 不使用PKE
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

            # 仍然需要kernel用于生成高斯热图（从初始标签）
            self.kernel = get_gaussian_kernel(kernlen=config['gaussian_kernel_size'],
                                              nsig=config['gaussian_sigma']).to(device)

        self.to(device)
    
    def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
        """
        不使用PKE的forward方法
        只使用初始标签计算损失，不学习额外关键点
        """
        detector_pred, descriptor_pred = self.network(x)
        
        if label_point_positions is not None:
            # 不使用PKE，只使用初始标签
            loss_detector_num = len(learn_index[0])
            loss_descriptor_num = loss_detector_num  # 描述子损失也只计算有标签的样本
            
            loss_detector = torch.tensor(0., requires_grad=True).to(x)
            loss_descriptor = torch.tensor(0., requires_grad=True).to(x)
            
            # 生成初始标签的高斯热图（不使用PKE学习）
            enhanced_label = None
            enhanced_label_pts = None
            
            if len(learn_index[0]) != 0:
                # 从初始标签生成高斯热图
                initial_label = F.conv2d(label_point_positions[learn_index], self.kernel,
                                       stride=1, padding=(self.kernel.shape[-1] - 1) // 2)
                initial_label[initial_label > 1] = 1
                
                # 计算检测器损失（只使用初始标签）
                loss_detector = self.dice(detector_pred[learn_index], initial_label.to(detector_pred))
                
                # 为了兼容训练工具的可视化，需要返回enhanced_label和enhanced_label_pts
                # 即使不使用PKE，也返回初始标签的高斯热图用于可视化
                enhanced_label_tmp = label_point_positions.clone()
                enhanced_label_tmp[learn_index] = initial_label
                enhanced_label = enhanced_label_tmp
                
                # enhanced_label_pts 使用初始标签点位置（不使用PKE学习到的额外点）
                enhanced_label_pts = label_point_positions.clone()
            
            # 计算描述子损失（需要affine变换用于一致性约束）
            with torch.no_grad():
                affine_x_for_desc, grid_for_desc, grid_inverse_for_desc = affine_images(x, used_for='descriptor')
                _, affine_descriptor_pred_for_desc = self.network(affine_x_for_desc)
            
            detector_pred_copy = detector_pred.clone().detach()
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(
                detector_pred_copy, label_point_positions,
                descriptor_pred,
                affine_descriptor_pred_for_desc,
                grid_inverse_for_desc
            )
            
            loss = loss_detector + loss_descriptor
            
            # 返回格式与原始SuperRetina兼容
            return loss, 0, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

        return detector_pred, descriptor_pred
    
    def load_pretrained_weights(self, checkpoint_path, device='cpu', strict=False):
        """
        安全加载预训练权重
        可以加载原始SuperRetina的权重（忽略PKE相关参数）
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取模型状态字典
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        else:
            pretrained_dict = checkpoint
        
        # 获取当前模型状态字典
        model_dict = self.state_dict()
        
        # 过滤掉不匹配的键（PKE相关参数会被忽略，因为模型结构相同）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新模型字典
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        # 统计信息
        total_params = len(model_dict)
        matched_params = len(pretrained_dict)
        
        print(f"Loaded pretrained weights from {checkpoint_path}")
        print(f"  Matched: {matched_params}/{total_params} parameters")
        print(f"  Note: This model does not use PKE module (PKE_learn=False)")
        
        return checkpoint


class SuperRetinaWithoutPKEWithAttention(SuperRetina):
    """
    不使用PKE模块 + 带自注意力机制的 SuperRetina 模型
    
    改进点：
    - 完全去掉PKE学习机制
    - 添加自注意力机制（在编码器高层）
    - 只使用初始标注的关键点进行训练
    - 简化训练流程，减少计算开销
    
    特点：
    - 完全向后兼容：可以加载原始 SuperRetina 的权重
    - 训练更简单：不需要value_map和affine变换用于PKE学习
    - 计算更快：减少了PKE相关的计算
    - 注意力增强：通过自注意力机制捕获长距离依赖
    - 只使用初始标签：直接使用标注的关键点，不学习额外关键点
    """
    def __init__(self, config=None, device='cpu', n_class=1, use_self_attention=True, attention_reduction=8):
        """
        Args:
            config: 配置字典
            device: 设备
            n_class: 输出类别数
            use_self_attention: 是否使用自注意力
            attention_reduction: 自注意力的降维比例（用于减少计算量）
        """
        # 先初始化父类（不调用super().__init__，因为我们要自定义）
        nn.Module.__init__(self)
        
        if not HAS_SELF_ATTENTION:
            raise ImportError("SelfAttention module not found. Please ensure attention_module.py contains SelfAttention class.")
        
        self.PKE_learn = False  # 不使用PKE
        self.use_self_attention = use_self_attention
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

        # 自注意力模块 - 只在编码器最高层使用（减少显存占用）
        # 只在conv4之后使用，因为：
        # 1. conv4分辨率最低（H/8），显存占用最小
        # 2. 高层特征语义更丰富，自注意力效果更好
        # 3. conv3分辨率较高（H/4），计算注意力矩阵会占用过多显存
        if self.use_self_attention:
            # 只在conv4使用自注意力，避免显存溢出
            self.self_attention4 = SelfAttention(c4, reduction=attention_reduction, use_residual=True)

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

            # 仍然需要kernel用于生成高斯热图（从初始标签）
            self.kernel = get_gaussian_kernel(kernlen=config['gaussian_kernel_size'],
                                              nsig=config['gaussian_sigma']).to(device)

        self.to(device)
    
    def network(self, x):
        """
        带自注意力的网络前向传播
        在编码器高层（conv4）应用自注意力
        """
        # 编码器第一层
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)
        
        # 编码器第二层
        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)
        
        # 编码器第三层
        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)
        
        # 编码器第四层 - 应用自注意力（分辨率最低，显存占用可接受）
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        if self.use_self_attention:
            x = self.self_attention4(x)

        # Descriptor Head - 与原始SuperRetina完全相同
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        desc = self.trans_conv(desc)

        # Detector Head - 与原始SuperRetina完全相同
        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)

        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        return semi, desc
    
    def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
        """
        不使用PKE的forward方法
        只使用初始标签计算损失，不学习额外关键点
        """
        detector_pred, descriptor_pred = self.network(x)
        
        if label_point_positions is not None:
            # 不使用PKE，只使用初始标签
            loss_detector_num = len(learn_index[0])
            loss_descriptor_num = loss_detector_num  # 描述子损失也只计算有标签的样本
            
            loss_detector = torch.tensor(0., requires_grad=True).to(x)
            loss_descriptor = torch.tensor(0., requires_grad=True).to(x)
            
            # 生成初始标签的高斯热图（不使用PKE学习）
            enhanced_label = None
            enhanced_label_pts = None
            
            if len(learn_index[0]) != 0:
                # 从初始标签生成高斯热图
                initial_label = F.conv2d(label_point_positions[learn_index], self.kernel,
                                       stride=1, padding=(self.kernel.shape[-1] - 1) // 2)
                initial_label[initial_label > 1] = 1
                
                # 计算检测器损失（只使用初始标签）
                loss_detector = self.dice(detector_pred[learn_index], initial_label.to(detector_pred))
                
                # 为了兼容训练工具的可视化，需要返回enhanced_label和enhanced_label_pts
                # 即使不使用PKE，也返回初始标签的高斯热图用于可视化
                enhanced_label_tmp = label_point_positions.clone()
                enhanced_label_tmp[learn_index] = initial_label
                enhanced_label = enhanced_label_tmp
                
                # enhanced_label_pts 使用初始标签点位置（不使用PKE学习到的额外点）
                enhanced_label_pts = label_point_positions.clone()
            
            # 计算描述子损失（需要affine变换用于一致性约束）
            with torch.no_grad():
                affine_x_for_desc, grid_for_desc, grid_inverse_for_desc = affine_images(x, used_for='descriptor')
                _, affine_descriptor_pred_for_desc = self.network(affine_x_for_desc)
            
            detector_pred_copy = detector_pred.clone().detach()
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(
                detector_pred_copy, label_point_positions,
                descriptor_pred,
                affine_descriptor_pred_for_desc,
                grid_inverse_for_desc
            )
            
            loss = loss_detector + loss_descriptor
            
            # 返回格式与原始SuperRetina兼容
            return loss, 0, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

        return detector_pred, descriptor_pred
    
    def load_pretrained_weights(self, checkpoint_path, device='cpu', strict=False):
        """
        安全加载预训练权重
        可以加载原始SuperRetina的权重（忽略PKE和自注意力相关参数）
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取模型状态字典
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        else:
            pretrained_dict = checkpoint
        
        # 获取当前模型状态字典
        model_dict = self.state_dict()
        
        # 过滤掉不匹配的键（PKE和自注意力相关参数会被忽略）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新模型字典
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        # 统计信息
        total_params = len(model_dict)
        matched_params = len(pretrained_dict)
        
        # 检查自注意力参数：当前模型中有，但预训练权重中没有的
        model_attention_params = [k for k in model_dict.keys() if 'self_attention' in k]
        pretrained_attention_params = [k for k in pretrained_dict.keys() if 'self_attention' in k]
        missing_attention_params = [k for k in model_attention_params if k not in pretrained_attention_params]
        
        print(f"Loaded pretrained weights from {checkpoint_path}")
        print(f"  Matched: {matched_params}/{total_params} parameters")
        if len(missing_attention_params) > 0:
            print(f"  Self-attention modules ({len(missing_attention_params)} params) are randomly initialized "
                  f"(not found in checkpoint)")
        elif len(model_attention_params) > 0:
            print(f"  Self-attention modules ({len(model_attention_params)} params) loaded from checkpoint")
        print(f"  Note: This model does not use PKE module (PKE_learn=False)")
        
        return checkpoint

class SuperRetinaWithPerceptualLoss(SuperRetina):
    """
    带感知损失（Perceptual Loss）的 SuperRetina 变体（0.05 单层 relu4_2 最佳版本）。
    - 仅新增单层 PerceptualLoss 模块
    - 保持原有 PKE、VALUE_MAP、descriptor_loss 逻辑完全不变
    - 支持 perceptual_start_epoch 延迟启用
    """

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        self.perceptual_loss = PerceptualLoss(device=device)
        self.perceptual_weight = config.get('perceptual_weight', 0.05) if config is not None else 0.05
        self.perceptual_start_epoch = config.get('perceptual_start_epoch', 0) if config is not None else 0
        self.current_epoch = 0
        print(
            f"✅ SuperRetinaWithPerceptualLoss 初始化完成，perceptual_weight={self.perceptual_weight}，"
            f"perceptual_start_epoch={self.perceptual_start_epoch}（单层 relu4_2）"
        )

    def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
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

            number_pts = 0
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

            # === 0.05 单层感知损失（支持延迟引入）===
            # 感知损失是否启用不应依赖 PKE；即使关闭 PKE，也可以在稳定若干轮后对有标签样本启用。
            if (
                len(learn_index[0]) != 0
                and hasattr(self, 'perceptual_loss')
                and self.current_epoch >= self.perceptual_start_epoch
            ):
                # affine_detector_pred is expressed in the affine-image coordinate
                # system; align it with detector_pred before comparing features.
                affine_pred_inverse = F.grid_sample(
                    affine_detector_pred[learn_index], grid_inverse[learn_index], align_corners=True
                )
                perc_input = affine_pred_inverse.repeat(1, 3, 1, 1)
                perc_target = detector_pred[learn_index].repeat(1, 3, 1, 1)
                perc_loss = self.perceptual_loss(perc_input, perc_target)
                loss_detector = loss_detector + self.perceptual_weight * perc_loss
                # print(f"Perceptual loss: {perc_loss.item():.4f} (weight={self.perceptual_weight})")  # 调试时可打开

            # 其余逻辑与父类完全一致
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
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(
                detector_pred_copy, label_point_positions,
                descriptor_pred, affine_descriptor_pred_for_desc, grid_inverse_for_desc)

            if self.PKE_learn and len(learn_index[0]) != 0:
                value_map[learn_index] = value_map_update

            loss = loss_detector + loss_descriptor
            return loss, number_pts, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

        return detector_pred, descriptor_pred

    def load_pretrained_weights(self, model_path, device=None, strict=False):
        if device is None:
            device = next(self.parameters()).device
        checkpoint = torch.load(model_path, map_location=device)
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        else:
            pretrained_dict = checkpoint
        model_dict = self.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(f"✅ Loaded SuperRetinaWithPerceptualLoss from {model_path} "
              f"(matched {len(filtered_dict)}/{len(pretrained_dict)} tensors)")

# class SuperRetinaWithPerceptualLoss(SuperRetina):
#     """
#     带感知损失（Perceptual Loss）的 SuperRetina 变体。
#     - 继承原始 SuperRetina 的全部结构（encoder + descriptor + detector）
#     - 仅新增 PerceptualLoss 模块（现已升级为 relu3_3 + relu4_2 多层加权）
#     - 保持原有 PKE、VALUE_MAP、descriptor_loss 等逻辑完全不变
#     """

#     def __init__(self, config=None, device='cpu', n_class=1):
#         # 先调用父类初始化（包含 encoder、descriptor head、detector head 等）
#         super().__init__(config=config, device=device, n_class=n_class)

#         # 新增：多层感知损失模块（relu3_3 + relu4_2 加权）
#         # 默认权重 0.4:0.6（可后续在 train.yaml 中配置 perceptual_layer_weights）
#         self.perceptual_loss = PerceptualLoss(device=device)

#         # 可通过 config 控制权重（推荐在 train.yaml 新增 perceptual_weight 字段）
#         self.perceptual_weight = config.get('perceptual_weight', 0.05) if config is not None else 0.05

#         print(f"✅ SuperRetinaWithPerceptualLoss 初始化完成，perceptual_weight={self.perceptual_weight} "
#               f"（多层 VGG: relu3_3 + relu4_2）")

#     def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
#         """
#         重写 forward，仅在 PKE_learn=True 且有 label 时加入感知损失
#         其他逻辑与父类完全一致
#         """
#         detector_pred, descriptor_pred = self.network(x)
#         enhanced_label_pts = None
#         enhanced_label = None

#         if label_point_positions is not None:
#             # === 原有 PKE + descriptor_loss 逻辑（完全复用父类代码）===
#             if self.PKE_learn:
#                 loss_detector_num = len(learn_index[0])
#                 loss_descriptor_num = x.shape[0]
#             else:
#                 loss_detector_num = len(learn_index[0])
#                 loss_descriptor_num = loss_detector_num

#             number_pts = 0
#             value_map_update = None
#             loss_detector = torch.tensor(0., requires_grad=True).to(x)
#             loss_descriptor = torch.tensor(0., requires_grad=True).to(x)

#             with torch.no_grad():
#                 affine_x, grid, grid_inverse = affine_images(x, used_for='detector')
#                 affine_detector_pred, affine_descriptor_pred = self.network(affine_x)

#             loss_cal = self.dice
#             if len(learn_index[0]) != 0:
#                 loss_detector, number_pts, value_map_update, enhanced_label_pts, enhanced_label = \
#                     pke_learn(detector_pred[learn_index], descriptor_pred[learn_index],
#                               grid_inverse[learn_index], affine_detector_pred[learn_index],
#                               affine_descriptor_pred[learn_index], self.kernel, loss_cal,
#                               label_point_positions[learn_index], value_map[learn_index],
#                               self.config, self.PKE_learn)

#             # === 感知损失（多层 VGG 版本，兼容新 perceptual_loss.py）===
#             if self.PKE_learn and len(learn_index[0]) != 0 and hasattr(self, 'perceptual_loss'):
#                 # 把单通道热图扩展为 3 通道（VGG 需要 RGB 输入）
#                 perc_input = affine_detector_pred.repeat(1, 3, 1, 1)
#                 perc_target = detector_pred[learn_index].repeat(1, 3, 1, 1)
#                 perc_loss = self.perceptual_loss(perc_input, perc_target)
#                 loss_detector = loss_detector + self.perceptual_weight * perc_loss

#                 # 可选打印调试信息（训练日志中会看到多层感知损失）
#                 print(f"Perceptual loss: {perc_loss.item():.4f} (weight={self.perceptual_weight})")

#             # 其余逻辑完全保持父类一致
#             if enhanced_label_pts is not None:
#                 enhanced_label_pts_tmp = label_point_positions.clone()
#                 enhanced_label_pts_tmp[learn_index] = enhanced_label_pts
#                 enhanced_label_pts = enhanced_label_pts_tmp
#             if enhanced_label is not None:
#                 enhanced_label_tmp = label_point_positions.clone()
#                 enhanced_label_tmp[learn_index] = enhanced_label
#                 enhanced_label = enhanced_label_tmp

#             detector_pred_copy = detector_pred.clone().detach()

#             affine_x_for_desc, grid_for_desc, grid_inverse_for_desc = affine_images(x, used_for='descriptor')
#             _, affine_descriptor_pred_for_desc = self.network(affine_x_for_desc)
#             loss_descriptor, descriptor_train_flag = self.descriptor_loss(
#                 detector_pred_copy, label_point_positions,
#                 descriptor_pred, affine_descriptor_pred_for_desc, grid_inverse_for_desc)

#             if self.PKE_learn and len(learn_index[0]) != 0:
#                 value_map[learn_index] = value_map_update

#             loss = loss_detector + loss_descriptor

#             return loss, number_pts, loss_detector.cpu().data.sum(), \
#                    loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
#                    enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

#         return detector_pred, descriptor_pred

#     def load_pretrained_weights(self, model_path, device=None, strict=False):
#         """兼容 SuperRetinaWithPerceptualLoss 的权重加载（与原 SuperRetina 一致）"""
#         if device is None:
#             device = next(self.parameters()).device
#         checkpoint = torch.load(model_path, map_location=device)
#         if 'net' in checkpoint:
#             pretrained_dict = checkpoint['net']
#         else:
#             pretrained_dict = checkpoint
#         model_dict = self.state_dict()
#         filtered_dict = {k: v for k, v in pretrained_dict.items()
#                          if k in model_dict and model_dict[k].shape == v.shape}
#         model_dict.update(filtered_dict)
#         self.load_state_dict(model_dict, strict=strict)
#         print(f"✅ Loaded SuperRetinaWithPerceptualLoss from {model_path} "
#               f"(matched {len(filtered_dict)}/{len(pretrained_dict)} tensors)")

class SuperRetinaWithVesselRegularization(SuperRetinaWithPerceptualLoss):
    """
    第一步优化版本：SuperJunction 风格血管正则化（基于 0.05 PerceptualLoss）
    - 继承当前最佳 WithPerceptualLoss（perceptual loss 完全不变）
    - 新增轻量 vessel head（使用 decoder cPa 特征）
    - 使用 enhanced_label 作为 pseudo vessel mask（auxiliary 为空时的替代）
    - 边界参数 border=8 完全不改动
    """

    def __init__(self, config=None, device='cpu', n_class=1):
        # === 关键修复：正确传递 device 给父类 ===
        super().__init__(config=config, device=device, n_class=n_class)

        # 新增：轻量 vessel head（使用 decoder cPa 特征）
        c1 = 64
        self.vessel_head = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 可通过 train.yaml 配置权重
        self.vessel_weight = config.get('vessel_weight', 0.3) if config is not None else 0.3

        # === 确保新模块也被迁移到正确设备 ===
        self.vessel_head.to(device)

        print(f"✅ SuperRetinaWithVesselRegularization 初始化完成，vessel_weight={self.vessel_weight}（使用 enhanced_label 作为 pseudo mask）")

    def network(self, x, return_cPa=False):
        """重写 network，返回 cPa 特征供 vessel head 使用（复用父类全部逻辑）"""
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)

        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)

        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Descriptor Head（完全不变）
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        desc = self.trans_conv(desc)

        # Detector Head（保留 cPa）
        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)
        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)
        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)
        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        if return_cPa:
            return semi, desc, cPa
        return semi, desc

    def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
        """重写 forward，加入 vessel regularization"""
        # 调用 network 时返回 cPa（仅训练阶段需要）
        if label_point_positions is not None:
            detector_pred, descriptor_pred, cPa = self.network(x, return_cPa=True)
        else:
            detector_pred, descriptor_pred = self.network(x)
            cPa = None

        enhanced_label_pts = None
        enhanced_label = None

        if label_point_positions is not None:
            if self.PKE_learn:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = x.shape[0]
            else:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = loss_detector_num

            number_pts = 0
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

            # === 0.05 单层感知损失（与 SuperRetinaWithPerceptualLoss 一致）===
            if (
                len(learn_index[0]) != 0
                and hasattr(self, 'perceptual_loss')
                and self.current_epoch >= self.perceptual_start_epoch
            ):
                # affine_detector_pred is expressed in the affine-image coordinate
                # system; align it with detector_pred before comparing features.
                affine_pred_inverse = F.grid_sample(
                    affine_detector_pred[learn_index], grid_inverse[learn_index], align_corners=True
                )
                perc_input = affine_pred_inverse.repeat(1, 3, 1, 1)
                perc_target = detector_pred[learn_index].repeat(1, 3, 1, 1)
                perc_loss = self.perceptual_loss(perc_input, perc_target)
                loss_detector = loss_detector + self.perceptual_weight * perc_loss

            # === 新增：vessel regularization（使用 enhanced_label 作为 pseudo vessel mask）===
            # enhanced_label 仅含 learn_index 子集，须对 cPa 同索引切片（与 vessel_only 一致）
            if cPa is not None and enhanced_label is not None and len(learn_index[0]) != 0:
                vessel_pred = self.vessel_head(cPa[learn_index])
                vessel_loss = loss_cal(vessel_pred, enhanced_label)
                loss_detector = loss_detector + self.vessel_weight * vessel_loss
                # print(f"Vessel loss: {vessel_loss.item():.4f} (weight={self.vessel_weight})")  # 调试时打开

            # === 其余逻辑完全与父类一致 ===
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
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(
                detector_pred_copy, label_point_positions,
                descriptor_pred, affine_descriptor_pred_for_desc, grid_inverse_for_desc)

            if self.PKE_learn and len(learn_index[0]) != 0:
                value_map[learn_index] = value_map_update

            loss = loss_detector + loss_descriptor

            return loss, number_pts, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

        return detector_pred, descriptor_pred

    def load_pretrained_weights(self, model_path, device=None, strict=False):
        """安全加载权重（兼容 predictor）"""
        if device is None:
            device = next(self.parameters()).device
        checkpoint = torch.load(model_path, map_location=device)
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        else:
            pretrained_dict = checkpoint
        model_dict = self.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(f"✅ Loaded SuperRetinaWithVesselRegularization from {model_path} "
              f"(matched {len(filtered_dict)}/{len(pretrained_dict)} tensors)")

class SuperRetinaWithVesselOnly(SuperRetina):
    """
    Ablation 变体：仅 vessel regularization（不带 perceptual loss）
    - 继承原始 SuperRetina 基类（完全干净）
    - 只新增轻量 vessel head（使用 decoder cPa 特征）
    - 使用 PKE 阶段的 enhanced_label 作为 pseudo vessel mask
    - 边界参数 border=8 完全不动
    """

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)

        # 新增：轻量 vessel head
        c1 = 64
        self.vessel_head = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 可配置权重（train.yaml 中设置）
        self.vessel_weight = config.get('vessel_weight', 0.3) if config is not None else 0.3

        self.vessel_head.to(device)

        print(f"✅ SuperRetinaWithVesselOnly 初始化完成，vessel_weight={self.vessel_weight}（纯 vessel regularization）")

    def network(self, x, return_cPa=False):
        """重写 network，返回 cPa 特征供 vessel head 使用（复用父类全部逻辑）"""
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)

        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)

        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Descriptor Head（完全不变）
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        desc = self.trans_conv(desc)

        # Detector Head（保留 cPa）
        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)
        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)
        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)
        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        if return_cPa:
            return semi, desc, cPa
        return semi, desc

    def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
        """重写 forward，只保留 vessel regularization 逻辑"""
        if label_point_positions is not None:
            detector_pred, descriptor_pred, cPa = self.network(x, return_cPa=True)
        else:
            detector_pred, descriptor_pred = self.network(x)
            cPa = None

        enhanced_label_pts = None
        enhanced_label = None

        if label_point_positions is not None:
            if self.PKE_learn:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = x.shape[0]
            else:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = loss_detector_num

            number_pts = 0
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

            # === 纯 vessel regularization（无 perceptual loss）===
            # enhanced_label 仅含 learn_index 子集，须对 cPa 同索引切片（与 pke_learn 输入一致）
            if cPa is not None and enhanced_label is not None and len(learn_index[0]) != 0:
                vessel_pred = self.vessel_head(cPa[learn_index])
                vessel_loss = loss_cal(vessel_pred, enhanced_label)
                loss_detector = loss_detector + self.vessel_weight * vessel_loss
                # print(f"Vessel loss: {vessel_loss.item():.4f} (weight={self.vessel_weight})")  # 调试时打开

            # === 其余逻辑与父类完全一致 ===
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
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(
                detector_pred_copy, label_point_positions,
                descriptor_pred, affine_descriptor_pred_for_desc, grid_inverse_for_desc)

            if self.PKE_learn and len(learn_index[0]) != 0:
                value_map[learn_index] = value_map_update

            loss = loss_detector + loss_descriptor

            return loss, number_pts, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

        return detector_pred, descriptor_pred

    def load_pretrained_weights(self, model_path, device=None, strict=False):
        """安全加载权重（兼容 predictor）"""
        if device is None:
            device = next(self.parameters()).device
        checkpoint = torch.load(model_path, map_location=device)
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        else:
            pretrained_dict = checkpoint
        model_dict = self.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(f"✅ Loaded SuperRetinaWithVesselOnly from {model_path} "
              f"(matched {len(filtered_dict)}/{len(pretrained_dict)} tensors)")


class SuperRetinaWithVesselOnlyMasked(SuperRetinaWithVesselOnly):
    """
    Phase 1A：区域选择性 vessel regularization。
    - 结构与 SuperRetinaWithVesselOnly 相同（推理路径一致）
    - vessel loss 仅在 online vessel mask 内计算，非血管区域不参与 vessel 梯度
    - target 仍为 PKE enhanced_label，与 mask 逐像素相乘后再做 MaskedDice
    """

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        self.masked_dice = MaskedDiceLoss()
        cfg = config or {}
        self.vessel_mask_backend = cfg.get('vessel_mask_backend', 'morph')
        self.vessel_mask_threshold = float(cfg.get('vessel_mask_threshold', 0.25))
        self.vessel_mask_dilate = int(cfg.get('vessel_mask_dilate', 3))
        self.stage1_epochs = int(cfg.get('vessel_stage1_epochs', 50))
        self.vessel_weight_stage2 = float(cfg.get('vessel_weight_stage2', 0.05))
        self.vessel_weight_floor = float(cfg.get('vessel_weight_floor', 0.0))
        self.stage2_decay = max(1, int(cfg.get('vessel_stage2_decay', 50)))
        # constant: fixed vessel_weight; three_stage: historical schedule;
        # continuous_to_floor: smoothly decays from vessel_weight to floor.
        self.vessel_schedule_mode = cfg.get('vessel_schedule_mode', 'three_stage')
        self.vessel_schedule_end_epoch = int(
            cfg.get('vessel_schedule_end_epoch', self.stage1_epochs + self.stage2_decay)
        )
        if self.vessel_schedule_mode not in {'constant', 'three_stage', 'continuous_to_floor'}:
            raise ValueError(
                f"Unknown vessel_schedule_mode: {self.vessel_schedule_mode}"
            )
        # PKE geometry admission can optionally follow a separate curriculum.
        # Defaults preserve every existing experiment: the configured threshold
        # remains fixed throughout training.
        self.pke_geometric_thresh = float(cfg.get('geometric_thresh', 0.5))
        self.pke_geometric_schedule_mode = cfg.get('pke_geometric_schedule_mode', 'constant')
        self.pke_geometric_relaxed_thresh = float(
            cfg.get('pke_geometric_relaxed_thresh', self.pke_geometric_thresh)
        )
        self.pke_geometric_relax_start_epoch = int(
            cfg.get('pke_geometric_relax_start_epoch', 0)
        )
        self.pke_geometric_relax_until_epoch = int(
            cfg.get('pke_geometric_relax_until_epoch', 0)
        )
        if self.pke_geometric_schedule_mode not in {
            'constant', 'relax_then_base', 'strict_relax_strict'
        }:
            raise ValueError(
                f"Unknown pke_geometric_schedule_mode: {self.pke_geometric_schedule_mode}"
            )
        if self.pke_geometric_schedule_mode in {
            'relax_then_base', 'strict_relax_strict'
        }:
            if not 0 <= self.pke_geometric_relaxed_thresh <= self.pke_geometric_thresh:
                raise ValueError(
                    'pke_geometric_relaxed_thresh must be in [0, geometric_thresh]'
                )
            if self.pke_geometric_relax_until_epoch <= 0:
                raise ValueError(
                    'pke_geometric_relax_until_epoch must be positive for relax_then_base'
                )
            if self.pke_geometric_schedule_mode == 'strict_relax_strict' and not (
                0 <= self.pke_geometric_relax_start_epoch < self.pke_geometric_relax_until_epoch
            ):
                raise ValueError(
                    'strict_relax_strict requires 0 <= relax_start_epoch < relax_until_epoch'
                )
        self.current_epoch = 0
        # PKE content matching defaults to the historical one-way criterion.
        # These validations make the G1 training options explicit while
        # leaving every older YAML fully compatible.
        self.pke_content_mode = cfg.get('pke_content_mode', 'one_way')
        self.pke_content_weak_feedback = bool(cfg.get('pke_content_weak_feedback', False))
        self.pke_content_strong_feedback_multiplier = int(
            cfg.get('pke_content_strong_feedback_multiplier', 1)
        )
        self.pke_content_weak_feedback_multiplier = int(
            cfg.get('pke_content_weak_feedback_multiplier', 1)
        )
        self.pke_descriptor_feedback_weight = float(cfg.get('pke_descriptor_feedback_weight', 0.0))
        self.pke_descriptor_margin = float(cfg.get('pke_descriptor_margin', 0.2))
        self.pke_descriptor_negative_distance = float(
            cfg.get('pke_descriptor_negative_distance', 16.0)
        )
        # G10: detector-independent descriptor supervision from the known
        # training affine. A zero weight preserves every historical run.
        self.dense_descriptor_weight = float(
            cfg.get('dense_descriptor_weight', 0.0)
        )
        self.dense_descriptor_ramp_epochs = int(
            cfg.get('dense_descriptor_ramp_epochs', 10)
        )
        self.dense_descriptor_grid_size = int(
            cfg.get('dense_descriptor_grid_size', 8)
        )
        self.dense_descriptor_structure_per_cell = int(
            cfg.get('dense_descriptor_structure_per_cell', 1)
        )
        self.dense_descriptor_uniform_per_cell = int(
            cfg.get('dense_descriptor_uniform_per_cell', 1)
        )
        self.dense_descriptor_border_margin = int(
            cfg.get('dense_descriptor_border_margin', 16)
        )
        self.dense_descriptor_valid_intensity = float(
            cfg.get('dense_descriptor_valid_intensity', 0.03)
        )
        self.dense_descriptor_min_negative_distance = float(
            cfg.get('dense_descriptor_min_negative_distance', 16.0)
        )
        self.dense_descriptor_margin = float(
            cfg.get('dense_descriptor_margin', 0.2)
        )
        self.log_dense_descriptor_stats = bool(
            cfg.get('log_dense_descriptor_stats', False)
        )
        if self.pke_content_mode not in {'one_way', 'bidirectional'}:
            raise ValueError(f'Unknown pke_content_mode: {self.pke_content_mode}')
        if (self.pke_content_strong_feedback_multiplier < 1
                or self.pke_content_weak_feedback_multiplier < 1):
            raise ValueError('PKE content feedback multipliers must be at least 1')
        if self.pke_descriptor_feedback_weight < 0 or self.pke_descriptor_margin < 0:
            raise ValueError('PKE descriptor feedback weight and margin must be non-negative')
        if self.dense_descriptor_weight < 0:
            raise ValueError('dense_descriptor_weight must be non-negative')
        if self.dense_descriptor_weight > 0 and (
            self.dense_descriptor_ramp_epochs < 1
            or self.dense_descriptor_grid_size < 1
            or self.dense_descriptor_structure_per_cell < 0
            or self.dense_descriptor_uniform_per_cell < 0
            or (
                self.dense_descriptor_structure_per_cell
                + self.dense_descriptor_uniform_per_cell
            ) < 1
            or self.dense_descriptor_border_margin < 0
            or not 0 <= self.dense_descriptor_valid_intensity <= 1
            or self.dense_descriptor_min_negative_distance < 0
            or self.dense_descriptor_margin < 0
        ):
            raise ValueError('Invalid balanced dense descriptor configuration')
        self.reset_dense_descriptor_epoch_stats()
        self.save_pke_diagnostics = bool(cfg.get('save_pke_diagnostics', False))
        self.pke_diagnostic_grid_size = int(cfg.get('pke_diagnostic_grid_size', 8))
        relaxed_threshold = cfg.get('pke_region_relaxed_threshold')
        self.pke_region_relaxed_threshold = (
            None if relaxed_threshold is None else float(relaxed_threshold)
        )
        if self.pke_region_relaxed_threshold is not None:
            base_threshold = float(cfg.get('geometric_thresh', 0.5))
            if not 0 <= self.pke_region_relaxed_threshold < base_threshold:
                raise ValueError(
                    'pke_region_relaxed_threshold must be non-negative and smaller than geometric_thresh'
                )
        # Optional G4-style ranking feedback. Defaults keep legacy PKE fully
        # unchanged: no second affine view exists until explicitly enabled.
        self.pke_multiview_noncore_feedback_enabled = bool(
            cfg.get('pke_multiview_noncore_feedback_enabled', False)
        )
        self.pke_multiview_noncore_start_epoch = int(
            cfg.get('pke_multiview_noncore_start_epoch', 10 ** 9)
        )
        self.pke_multiview_noncore_bonus = int(
            cfg.get('pke_multiview_noncore_bonus', 0)
        )
        self.pke_multiview_noncore_grid_size = int(
            cfg.get('pke_multiview_noncore_grid_size', 8)
        )
        self.pke_multiview_noncore_border_margin = int(
            cfg.get('pke_multiview_noncore_border_margin', 48)
        )
        self.pke_multiview_noncore_low_density_max = int(
            cfg.get('pke_multiview_noncore_low_density_max', 4)
        )
        self.pke_multiview_noncore_max_per_image = int(
            cfg.get('pke_multiview_noncore_max_per_image', 8)
        )
        if self.pke_multiview_noncore_start_epoch < 0:
            raise ValueError('pke_multiview_noncore_start_epoch must be non-negative')
        if (self.pke_multiview_noncore_bonus < 0
                or self.pke_multiview_noncore_grid_size < 2
                or self.pke_multiview_noncore_border_margin < 0
                or self.pke_multiview_noncore_low_density_max < 0
                or self.pke_multiview_noncore_max_per_image < 0):
            raise ValueError('Invalid multiview non-core feedback configuration')
        self.last_pke_diagnostics = None
        print(
            f"✅ SuperRetinaWithVesselOnlyMasked 初始化完成，"
            f"vessel_weight={self.vessel_weight}，"
            f"mask_backend={self.vessel_mask_backend}，"
            f"threshold={self.vessel_mask_threshold}，"
            f"stage1_epochs={self.stage1_epochs}，"
            f"stage2_vessel_weight={self.vessel_weight_stage2}，"
            f"schedule={self.vessel_schedule_mode}，"
            f"pke_geometry={self.pke_geometric_schedule_mode}"
        )

    def _build_vessel_masks(self, images):
        return compute_vessel_mask_batch(
            images,
            backend=self.vessel_mask_backend,
            threshold=self.vessel_mask_threshold,
            dilate_kernel=self.vessel_mask_dilate,
        )

    def _get_stage_vessel_weight(self):
        if self.vessel_schedule_mode == 'constant':
            return self.vessel_weight

        if self.current_epoch < self.stage1_epochs:
            return self.vessel_weight

        if self.vessel_schedule_mode == 'continuous_to_floor':
            total_decay = max(1, self.vessel_schedule_end_epoch - self.stage1_epochs)
            progress = min(
                (self.current_epoch - self.stage1_epochs) / float(total_decay), 1.0
            )
            return self.vessel_weight + progress * (
                self.vessel_weight_floor - self.vessel_weight
            )

        stage2_step = self.current_epoch - self.stage1_epochs
        if stage2_step >= self.stage2_decay:
            return self.vessel_weight_floor
        progress = stage2_step / float(self.stage2_decay)
        return self.vessel_weight_stage2 + (self.vessel_weight - self.vessel_weight_stage2) * (1.0 - progress)

    def _get_pke_geometric_thresh(self):
        """Return the PKE admission threshold for the current training epoch."""
        if (
            self.pke_geometric_schedule_mode == 'relax_then_base'
            and self.current_epoch < self.pke_geometric_relax_until_epoch
        ):
            return self.pke_geometric_relaxed_thresh
        if (
            self.pke_geometric_schedule_mode == 'strict_relax_strict'
            and self.pke_geometric_relax_start_epoch <= self.current_epoch
            < self.pke_geometric_relax_until_epoch
        ):
            return self.pke_geometric_relaxed_thresh
        return self.pke_geometric_thresh

    def _multiview_noncore_feedback_active(self):
        return (
            self.pke_multiview_noncore_feedback_enabled
            and self.pke_multiview_noncore_bonus > 0
            and self.current_epoch >= self.pke_multiview_noncore_start_epoch
        )

    def reset_dense_descriptor_epoch_stats(self):
        self._dense_descriptor_epoch_stats = {
            'calls': 0,
            'sampled_points': 0,
            'valid_pairs': 0,
            'occupied_cells': 0,
            'positive_distance': 0.0,
            'negative_distance': 0.0,
            'loss': 0.0,
        }

    def dense_descriptor_epoch_summary(self):
        stats = self._dense_descriptor_epoch_stats
        calls = max(1, stats['calls'])
        pairs = max(1, stats['valid_pairs'])
        return {
            'calls': float(stats['calls']),
            'sampled_points_per_call': stats['sampled_points'] / calls,
            'valid_pairs_per_call': stats['valid_pairs'] / calls,
            'occupied_cells_per_call': stats['occupied_cells'] / calls,
            'positive_distance': stats['positive_distance'] / pairs,
            'negative_distance': stats['negative_distance'] / pairs,
            'loss': stats['loss'] / calls,
        }

    def _balanced_dense_descriptor_loss(
            self, images, descriptor_pred, affine_descriptor_pred,
            grid_inverse):
        """Grid-balanced affine correspondences independent of detector output."""
        if self.dense_descriptor_weight <= 0:
            return descriptor_pred.sum() * 0
        batch_size, _, height, width = images.shape
        grid_size = self.dense_descriptor_grid_size
        if height % grid_size or width % grid_size:
            raise ValueError(
                'dense_descriptor_grid_size must divide image height and width'
            )
        cell_height = height // grid_size
        cell_width = width // grid_size
        per_cell = (
            self.dense_descriptor_structure_per_cell
            + self.dense_descriptor_uniform_per_cell
        )
        with torch.no_grad():
            gray = images.mean(dim=1, keepdim=True)
            sobel_x = gray.new_tensor(
                [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]
            ).unsqueeze(0)
            sobel_y = sobel_x.transpose(-1, -2)
            gradient = (
                F.conv2d(gray, sobel_x, padding=1).square()
                + F.conv2d(gray, sobel_y, padding=1).square()
            ).sqrt().squeeze(1)
            valid = gray.squeeze(1) >= self.dense_descriptor_valid_intensity
            margin = self.dense_descriptor_border_margin
            if margin:
                valid[:, :margin] = False
                valid[:, -margin:] = False
                valid[:, :, :margin] = False
                valid[:, :, -margin:] = False
            # [B, grid_y, grid_x, cell_y, cell_x] -> [B, cells, pixels]
            gradient_cells = gradient.reshape(
                batch_size, grid_size, cell_height, grid_size, cell_width
            ).permute(0, 1, 3, 2, 4).reshape(
                batch_size, grid_size * grid_size, -1
            )
            valid_cells = valid.reshape(
                batch_size, grid_size, cell_height, grid_size, cell_width
            ).permute(0, 1, 3, 2, 4).reshape_as(gradient_cells)
            chosen = torch.zeros_like(valid_cells)
            selected_parts = []
            if self.dense_descriptor_structure_per_cell:
                scores = gradient_cells.masked_fill(~valid_cells, float('-inf'))
                structure = scores.topk(
                    self.dense_descriptor_structure_per_cell, dim=-1
                ).indices
                selected_parts.append(structure)
                chosen.scatter_(-1, structure, True)
            if self.dense_descriptor_uniform_per_cell:
                uniform_scores = torch.rand_like(gradient_cells).masked_fill(
                    ~valid_cells | chosen, float('-inf')
                )
                uniform = uniform_scores.topk(
                    self.dense_descriptor_uniform_per_cell, dim=-1
                ).indices
                selected_parts.append(uniform)
            selected = torch.cat(selected_parts, dim=-1)

        losses = []
        stats = self._dense_descriptor_epoch_stats
        for batch_index in range(batch_size):
            cell_ids = torch.arange(
                grid_size * grid_size, device=images.device
            )[:, None].expand(-1, per_cell).reshape(-1)
            local_ids = selected[batch_index].reshape(-1)
            ys = (
                torch.div(cell_ids, grid_size, rounding_mode='floor')
                * cell_height
                + torch.div(local_ids, cell_width, rounding_mode='floor')
            )
            xs = cell_ids.remainder(grid_size) * cell_width + local_ids.remainder(
                cell_width
            )
            source_valid = valid[batch_index, ys, xs]
            mapped = grid_inverse[batch_index, ys, xs]
            mapped_valid = (mapped.abs() < 1).all(dim=1)
            keep = source_valid & mapped_valid
            if keep.sum() < 2:
                continue
            xs, ys, mapped = xs[keep], ys[keep], mapped[keep]
            source_points = torch.stack((xs, ys), dim=1).float()
            affine_points = torch.stack((
                (mapped[:, 0] + 1) * 0.5 * (width - 1),
                (mapped[:, 1] + 1) * 0.5 * (height - 1),
            ), dim=1)
            anchor = sample_keypoint_desc(
                source_points[None],
                descriptor_pred[batch_index:batch_index + 1], s=8
            )[0].T
            positive = sample_keypoint_desc(
                affine_points[None],
                affine_descriptor_pred[batch_index:batch_index + 1], s=8
            )[0].T
            distances = torch.cdist(anchor, positive)
            spatial = torch.cdist(
                source_points, source_points
            ) >= self.dense_descriptor_min_negative_distance
            negative = distances.masked_fill(~spatial, float('inf')).min(
                dim=1
            ).values
            positive_distance = distances.diag()
            pair_valid = torch.isfinite(negative)
            if not pair_valid.any():
                continue
            image_loss = F.relu(
                positive_distance[pair_valid] - negative[pair_valid]
                + self.dense_descriptor_margin
            ).mean()
            losses.append(image_loss)
            if self.log_dense_descriptor_stats:
                valid_count = int(pair_valid.sum().item())
                stats['sampled_points'] += int(keep.sum().item())
                stats['valid_pairs'] += valid_count
                stats['occupied_cells'] += int(
                    torch.unique(cell_ids[keep]).numel()
                )
                stats['positive_distance'] += float(
                    positive_distance[pair_valid].sum().detach().item()
                )
                stats['negative_distance'] += float(
                    negative[pair_valid].sum().detach().item()
                )
        result = (
            torch.stack(losses).mean()
            if losses else descriptor_pred.sum() * 0
        )
        if self.log_dense_descriptor_stats:
            stats['calls'] += 1
            stats['loss'] += float(result.detach().item())
        return result

    def _pke_descriptor_feedback_loss(self, detector_pred, descriptor_pred, grid_inverse,
                                      affine_detector_pred, affine_descriptor_pred, affine_x, learn_index):
        """Directly train descriptors from bidirectionally verified PKE pairs."""
        if self.pke_descriptor_feedback_weight <= 0 or len(learn_index[0]) == 0:
            return descriptor_pred.sum() * 0
        with torch.no_grad():
            candidates = nms(detector_pred[learn_index], nms_thresh=self.nms_thresh,
                             nms_size=self.nms_size)
            points, affine_points = mapping_points(grid_inverse[learn_index], candidates,
                                                   detector_pred.shape[-2], detector_pred.shape[-1])
            geo, affine_geo = geometric_filter(affine_detector_pred[learn_index], points, affine_points,
                                               geometric_thresh=self._get_pke_geometric_thresh())
            pairs, affine_pairs = content_filter(descriptor_pred[learn_index].detach(),
                                                 affine_descriptor_pred[learn_index].detach(),
                                                 geo, affine_geo, mode='bidirectional')
        # Recompute the affine descriptor branch with gradients; pair selection above is detached.
        _, affine_desc = self.network(affine_x)
        losses = []
        for local, (pts, aff_pts) in enumerate(zip(pairs, affine_pairs)):
            if not torch.is_tensor(pts) or len(pts) < 2:
                continue
            anchor = sample_keypoint_desc(pts[None], descriptor_pred[learn_index][local:local + 1], s=8)[0].T
            positive = sample_keypoint_desc(aff_pts[None], affine_desc[learn_index][local:local + 1], s=8)[0].T
            distances = torch.cdist(anchor, positive)
            spatial = torch.cdist(pts.float(), pts.float()) >= self.pke_descriptor_negative_distance
            inf = torch.full_like(distances, float('inf'))
            neg_q = torch.where(spatial, distances, inf).min(dim=1).values
            neg_r = torch.where(spatial, distances, inf).min(dim=0).values
            pos = distances.diag()
            valid = torch.isfinite(neg_q) & torch.isfinite(neg_r)
            if valid.any():
                losses.append((F.relu(pos[valid] - neg_q[valid] + self.pke_descriptor_margin).mean() +
                               F.relu(pos[valid] - neg_r[valid] + self.pke_descriptor_margin).mean()) * 0.5)
        return torch.stack(losses).mean() if losses else descriptor_pred.sum() * 0

    def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
        self.last_pke_diagnostics = None
        if label_point_positions is not None:
            detector_pred, descriptor_pred, cPa = self.network(x, return_cPa=True)
        else:
            detector_pred, descriptor_pred = self.network(x)
            cPa = None

        enhanced_label_pts = None
        enhanced_label = None

        if label_point_positions is not None:
            if self.PKE_learn:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = x.shape[0]
            else:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = loss_detector_num

            number_pts = 0
            value_map_update = None
            loss_detector = torch.tensor(0., requires_grad=True).to(x)
            loss_descriptor = torch.tensor(0., requires_grad=True).to(x)

            with torch.no_grad():
                affine_x, grid, grid_inverse = affine_images(x, used_for='detector')
                affine_detector_pred, affine_descriptor_pred = self.network(affine_x)
                multiview_active = self._multiview_noncore_feedback_active()
                stability_grid_inverse = None
                stability_affine_detector_pred = None
                stability_affine_descriptor_pred = None
                if multiview_active:
                    stability_affine_x, _, stability_grid_inverse = affine_images(
                        x, used_for='detector'
                    )
                    stability_affine_detector_pred, stability_affine_descriptor_pred = self.network(
                        stability_affine_x
                    )
                vessel_mask = (
                    self._build_vessel_masks(x)
                    if self.pke_region_relaxed_threshold is not None or multiview_active else None
                )
                ema_teacher_active = (
                    getattr(self, 'pke_ema_teacher_enabled', False)
                    and self.current_epoch >= getattr(
                        self, 'pke_ema_teacher_start_epoch', 0
                    )
                )
                teacher_detector_pred = None
                teacher_descriptor_pred = None
                teacher_affine_detector_pred = None
                teacher_affine_descriptor_pred = None
                if ema_teacher_active:
                    # The teacher only chooses pseudo-labels. Student tensors
                    # remain in the detector losses below, so no student
                    # gradient is cut by this no_grad branch.
                    self.ema_teacher.eval()
                    teacher_detector_pred, teacher_descriptor_pred = (
                        self.ema_teacher.network(x)
                    )
                    (
                        teacher_affine_detector_pred,
                        teacher_affine_descriptor_pred,
                    ) = self.ema_teacher.network(affine_x)

            loss_cal = self.dice
            pke_stage_points = None
            if len(learn_index[0]) != 0:
                # Never mutate self.config: the per-step override is isolated
                # to this PKE call and leaves legacy configurations unchanged.
                pke_config = dict(self.config)
                pke_config['geometric_thresh'] = self._get_pke_geometric_thresh()
                pke_config['pke_multiview_noncore_feedback_active'] = multiview_active
                pke_result = pke_learn(
                    detector_pred[learn_index], descriptor_pred[learn_index],
                    grid_inverse[learn_index], affine_detector_pred[learn_index],
                    affine_descriptor_pred[learn_index], self.kernel, loss_cal,
                    label_point_positions[learn_index], value_map[learn_index],
                    pke_config, self.PKE_learn,
                    return_stage_points=self.save_pke_diagnostics,
                    vessel_masks=None if vessel_mask is None else vessel_mask[learn_index],
                    relaxed_non_core_thresh=self.pke_region_relaxed_threshold,
                    stability_grid_inverse=(
                        None if stability_grid_inverse is None else stability_grid_inverse[learn_index]
                    ),
                    stability_affine_detector_pred=(
                        None if stability_affine_detector_pred is None
                        else stability_affine_detector_pred[learn_index]
                    ),
                    stability_affine_descriptor_pred=(
                        None if stability_affine_descriptor_pred is None
                        else stability_affine_descriptor_pred[learn_index]
                    ),
                    candidate_detector_pred=(
                        None if teacher_detector_pred is None
                        else teacher_detector_pred[learn_index]
                    ),
                    validation_affine_detector_pred=(
                        None if teacher_affine_detector_pred is None
                        else teacher_affine_detector_pred[learn_index]
                    ),
                    validation_descriptor_pred=(
                        None if teacher_descriptor_pred is None
                        else teacher_descriptor_pred[learn_index]
                    ),
                    validation_affine_descriptor_pred=(
                        None if teacher_affine_descriptor_pred is None
                        else teacher_affine_descriptor_pred[learn_index]
                    ),
                )
                if self.save_pke_diagnostics:
                    loss_detector, number_pts, value_map_update, enhanced_label_pts, enhanced_label, pke_stage_points = pke_result
                else:
                    loss_detector, number_pts, value_map_update, enhanced_label_pts, enhanced_label = pke_result

            if cPa is not None and enhanced_label is not None and len(learn_index[0]) != 0:
                with torch.no_grad():
                    if vessel_mask is None:
                        vessel_mask = self._build_vessel_masks(x)
                vessel_pred = self.vessel_head(cPa[learn_index])
                # enhanced_label from pke_learn is already aligned with learn_index batch
                vessel_mask_sub = vessel_mask[learn_index]
                vessel_loss = self.masked_dice(vessel_pred, enhanced_label, vessel_mask_sub)
                vessel_weight = self._get_stage_vessel_weight()
                loss_detector = loss_detector + vessel_weight * vessel_loss

            if self.save_pke_diagnostics and pke_stage_points is not None:
                with torch.no_grad():
                    if vessel_mask is None:
                        vessel_mask = self._build_vessel_masks(x)
                    self.last_pke_diagnostics = summarize_pke_stages(
                        pke_stage_points, vessel_mask[learn_index], x.shape[-2:],
                        grid_size=self.pke_diagnostic_grid_size,
                    )

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
            if getattr(self, '_supports_descriptor_only_forward', False):
                affine_descriptor_pred_for_desc = self.network(
                    affine_x_for_desc, descriptor_only=True
                )
            else:
                _, affine_descriptor_pred_for_desc = self.network(
                    affine_x_for_desc
                )
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(
                detector_pred_copy, label_point_positions,
                descriptor_pred, affine_descriptor_pred_for_desc, grid_inverse_for_desc)
            dense_descriptor_loss = self._balanced_dense_descriptor_loss(
                x, descriptor_pred, affine_descriptor_pred_for_desc,
                grid_inverse_for_desc,
            )
            dense_ramp = min(
                1.0,
                (self.current_epoch + 1)
                / float(max(1, self.dense_descriptor_ramp_epochs)),
            )
            loss_descriptor = (
                loss_descriptor
                + self.dense_descriptor_weight
                * dense_ramp
                * dense_descriptor_loss
            )
            pke_descriptor_loss = self._pke_descriptor_feedback_loss(
                detector_pred, descriptor_pred, grid_inverse, affine_detector_pred,
                affine_descriptor_pred, affine_x, learn_index,
            )
            loss_descriptor = loss_descriptor + self.pke_descriptor_feedback_weight * pke_descriptor_loss

            if self.PKE_learn and len(learn_index[0]) != 0:
                value_map[learn_index] = value_map_update

            # Read-only diagnostics may inspect the two live graphs separately.
            # The attribute is absent by default, so every historical training and
            # inference path keeps the exact same return value and gradient behavior.
            if getattr(self, '_capture_gradient_audit_losses', False):
                self._gradient_audit_losses = (loss_detector, loss_descriptor)

            loss = loss_detector + loss_descriptor

            return loss, number_pts, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

        return detector_pred, descriptor_pred

    def load_pretrained_weights(self, model_path, device=None, strict=False):
        if device is None:
            device = next(self.parameters()).device
        checkpoint = torch.load(model_path, map_location=device)
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        else:
            pretrained_dict = checkpoint
        model_dict = self.state_dict()
        filtered_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(
            f"✅ Loaded SuperRetinaWithVesselOnlyMasked from {model_path} "
            f"(matched {len(filtered_dict)}/{len(pretrained_dict)} tensors)"
        )


class SuperRetinaWithDecoupledMultiScaleDescriptor(SuperRetinaWithVesselOnlyMasked):
    """G6: G0 detector with a deep-decoupled, multi-scale descriptor encoder.

    conv1-conv2 remain shared. The detector keeps the historical conv3-conv4
    path, while the descriptor owns independent conv3-conv4 layers and fuses
    shared shallow detail, descriptor mid-level, and descriptor deep features.
    """

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        c2, c3, c4, descriptor_channels = 64, 128, 128, 256

        self.descriptor_conv3a = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.descriptor_conv3b = nn.Conv2d(c3, c3, kernel_size=3, padding=1)
        self.descriptor_conv4a = nn.Conv2d(c3, c4, kernel_size=3, padding=1)
        self.descriptor_conv4b = nn.Conv2d(c4, c4, kernel_size=3, padding=1)

        self.descriptor_shallow_projection = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.descriptor_mid_projection = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.descriptor_multiscale_fusion = nn.Sequential(
            nn.Conv2d(c2 + c3 + c4, descriptor_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # G6 replaces the legacy single-scale convDa path entirely.
        del self.convDa
        self.to(device)
        print(
            '✅ SuperRetinaWithDecoupledMultiScaleDescriptor 初始化完成，'
            '共享层=conv1-conv2，独立描述子层=conv3-conv4，多尺度融合=conv2/3/4'
        )

    def network(self, x, return_cPa=False):
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)

        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        shared_after_conv2 = self.pool(conv2)

        # Detector-specific deep encoder; this is the historical G0 path.
        detector_x = self.relu(self.conv3a(shared_after_conv2))
        detector_conv3 = self.relu(self.conv3b(detector_x))
        detector_x = self.pool(detector_conv3)
        detector_x = self.relu(self.conv4a(detector_x))
        detector_conv4 = self.relu(self.conv4b(detector_x))

        # Descriptor-specific deep encoder.
        descriptor_x = self.relu(self.descriptor_conv3a(shared_after_conv2))
        descriptor_conv3 = self.relu(self.descriptor_conv3b(descriptor_x))
        descriptor_x = self.pool(descriptor_conv3)
        descriptor_x = self.relu(self.descriptor_conv4a(descriptor_x))
        descriptor_conv4 = self.relu(self.descriptor_conv4b(descriptor_x))

        descriptor_shallow = self.descriptor_shallow_projection(conv2)
        descriptor_mid = self.descriptor_mid_projection(descriptor_conv3)
        descriptor_fused = self.descriptor_multiscale_fusion(torch.cat(
            [descriptor_shallow, descriptor_mid, descriptor_conv4], dim=1
        ))
        cDb = self.relu(self.convDb(descriptor_fused))
        desc = self.convDc(cDb)
        desc = F.normalize(desc, p=2, dim=1)
        desc = self.trans_conv(desc)

        cPa = self.upsample(detector_conv4)
        cPa = torch.cat([cPa, detector_conv3], dim=1)
        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)
        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)
        cPa = self.dconv_up1(cPa)
        semi = torch.sigmoid(self.conv_last(cPa))

        if return_cPa:
            return semi, desc, cPa
        return semi, desc


class SuperRetinaWithResidualMultiScaleDescriptor(SuperRetinaWithVesselOnlyMasked):
    """G7: preserve the G0 descriptor path and add gated multi-scale residual detail.

    Unlike G6, the complete shared conv1-conv4 encoder and legacy convDa descriptor
    path remain intact. Features from conv2 and conv3 are projected to the conv4
    resolution, fused, and added to convDa through a learnable scalar gate. This
    makes the initial architecture close to G0 while allowing training to use
    shallow and mid-level retinal detail when it is beneficial.
    """

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        c2, c3, descriptor_channels = 64, 128, 256
        gate_init = 0.1
        if config is not None:
            gate_init = float(config.get('descriptor_multiscale_gate_init', 0.1))
        if not 0.0 < gate_init < 1.0:
            raise ValueError(
                'descriptor_multiscale_gate_init must be strictly between 0 and 1'
            )

        self.descriptor_residual_shallow = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.descriptor_residual_mid = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.descriptor_residual_fusion = nn.Sequential(
            nn.Conv2d(c2 + c3, descriptor_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.descriptor_multiscale_gate_logit = nn.Parameter(
            torch.tensor(gate_logit, dtype=torch.float32)
        )
        self.log_descriptor_gate_stats = bool(
            config.get('log_descriptor_gate_stats', False)
            if config is not None else False
        )
        self.descriptor_gate_stats_max_calls = int(
            config.get('descriptor_gate_stats_max_calls', 64)
            if config is not None else 64
        )
        if self.descriptor_gate_stats_max_calls <= 0:
            raise ValueError(
                'descriptor_gate_stats_max_calls must be positive'
            )
        self.reset_descriptor_gate_epoch_stats()
        self._supports_descriptor_only_forward = True
        self.to(device)
        print(
            '✅ SuperRetinaWithResidualMultiScaleDescriptor 初始化完成，'
            f'G0描述子主路径保留，多尺度残差=conv2/3，初始门控={gate_init:.4f}'
        )

    def reset_descriptor_gate_epoch_stats(self):
        self._descriptor_gate_epoch_stats = {
            'calls': 0,
            'gate_mean': 0.0,
            'gate_std': 0.0,
            'gate_p10': 0.0,
            'gate_p50': 0.0,
            'gate_p90': 0.0,
            'near_upper_fraction': 0.0,
            'main_norm': 0.0,
            'residual_norm': 0.0,
            'injection_ratio': 0.0,
            'agreement_mean': 0.0,
            'negative_agreement_fraction': 0.0,
            'agreement_calls': 0,
        }

    def descriptor_gate_epoch_summary(self):
        stats = self._descriptor_gate_epoch_stats
        calls = max(1, stats['calls'])
        summary = {
            key: stats[key] / calls
            for key in (
                'gate_mean', 'gate_std', 'gate_p10', 'gate_p50', 'gate_p90',
                'near_upper_fraction', 'main_norm', 'residual_norm',
                'injection_ratio',
            )
        }
        summary['sampled_calls'] = float(stats['calls'])
        agreement_calls = stats['agreement_calls']
        if agreement_calls:
            summary.update({
                'agreement_mean':
                    stats['agreement_mean'] / agreement_calls,
                'negative_agreement_fraction':
                    stats['negative_agreement_fraction'] / agreement_calls,
            })
        return summary

    def _record_descriptor_gate_stats(
        self, gate, descriptor_main, descriptor_residual, agreement=None
    ):
        if not self.training or not self.log_descriptor_gate_stats:
            return
        if (
            self._descriptor_gate_epoch_stats['calls']
            >= self.descriptor_gate_stats_max_calls
        ):
            return
        with torch.no_grad():
            flat = gate.detach().reshape(-1)
            stride = max(1, flat.numel() // 1024)
            sample = flat[::stride][:1024].float()
            quantiles = torch.quantile(
                sample, torch.tensor(
                    [0.1, 0.5, 0.9], device=sample.device
                )
            )
            gate_upper = float(getattr(self, 'descriptor_gate_max', 1.0))
            main_norm = torch.norm(
                descriptor_main.detach(), p=2, dim=1
            )
            residual_norm = torch.norm(
                descriptor_residual.detach(), p=2, dim=1
            )
            injected_norm = torch.norm(
                gate.detach() * descriptor_residual.detach(), p=2, dim=1
            )
            stats = self._descriptor_gate_epoch_stats
            stats['calls'] += 1
            stats['gate_mean'] += float(sample.mean().item())
            stats['gate_std'] += float(sample.std(unbiased=False).item())
            stats['gate_p10'] += float(quantiles[0].item())
            stats['gate_p50'] += float(quantiles[1].item())
            stats['gate_p90'] += float(quantiles[2].item())
            stats['near_upper_fraction'] += float(
                (sample >= 0.9 * gate_upper).float().mean().item()
            )
            stats['main_norm'] += float(main_norm.mean().item())
            stats['residual_norm'] += float(residual_norm.mean().item())
            stats['injection_ratio'] += float(
                (injected_norm / main_norm.clamp_min(1e-6)).mean().item()
            )
            if agreement is not None:
                agreement_sample = agreement.detach().reshape(-1)
                agreement_stride = max(
                    1, agreement_sample.numel() // 1024
                )
                agreement_sample = agreement_sample[
                    ::agreement_stride
                ][:1024].float()
                stats['agreement_calls'] += 1
                stats['agreement_mean'] += float(
                    agreement_sample.mean().item()
                )
                stats['negative_agreement_fraction'] += float(
                    (agreement_sample < 0).float().mean().item()
                )

    def _descriptor_gate(
        self, descriptor_main, descriptor_residual
    ):
        gate = torch.sigmoid(self.descriptor_multiscale_gate_logit)
        return gate, None

    def _descriptor_injection(
        self, descriptor_main, descriptor_residual
    ):
        """Return the residual injected into the legacy descriptor path.

        Keeping this operation behind a hook lets newer variants constrain the
        injection without changing G7/G15 numerics or their checkpoints.
        """
        descriptor_gate, descriptor_agreement = self._descriptor_gate(
            descriptor_main, descriptor_residual
        )
        self._record_descriptor_gate_stats(
            descriptor_gate,
            descriptor_main,
            descriptor_residual,
            descriptor_agreement,
        )
        return descriptor_gate * descriptor_residual

    def network(self, x, return_cPa=False, descriptor_only=False):
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)

        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)

        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)

        x = self.relu(self.conv4a(x))
        conv4 = self.relu(self.conv4b(x))

        descriptor_main = self.relu(self.convDa(conv4))
        descriptor_residual = self.descriptor_residual_fusion(torch.cat(
            [
                self.descriptor_residual_shallow(conv2),
                self.descriptor_residual_mid(conv3),
            ],
            dim=1,
        ))
        descriptor_injection = self._descriptor_injection(
            descriptor_main, descriptor_residual
        )
        cDb = self.relu(self.convDb(
            descriptor_main + descriptor_injection
        ))
        desc = self.convDc(cDb)
        descriptor_norm = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(descriptor_norm, 1))
        desc = self.trans_conv(desc)
        if descriptor_only:
            return desc

        cPa = self.upsample(conv4)
        cPa = torch.cat([cPa, conv3], dim=1)
        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)
        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)
        cPa = self.dconv_up1(cPa)
        semi = torch.sigmoid(self.conv_last(cPa))

        if return_cPa:
            return semi, desc, cPa
        return semi, desc


class SuperRetinaWithZeroStartResidualMultiScaleDescriptor(
    SuperRetinaWithResidualMultiScaleDescriptor
):
    """G15: G7 multi-scale detail with an exactly zero initial residual."""

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        c2, c3, descriptor_channels = 64, 128, 256
        self.descriptor_residual_fusion = nn.Sequential(
            nn.Conv2d(
                c2 + c3, descriptor_channels,
                kernel_size=3, padding=1,
            ),
            nn.ReLU(inplace=True),
            # No activation follows this projection: at zero initialization
            # its weights receive gradients on the first optimizer step.
            nn.Conv2d(
                descriptor_channels, descriptor_channels, kernel_size=1
            ),
        )
        nn.init.zeros_(self.descriptor_residual_fusion[-1].weight)
        nn.init.zeros_(self.descriptor_residual_fusion[-1].bias)
        self.to(device)
        print(
            'SuperRetinaWithZeroStartResidualMultiScaleDescriptor '
            'initialized: descriptor residual projection starts at zero'
        )


class SuperRetinaWithNormControlledZeroStartMultiScaleDescriptor(
    SuperRetinaWithZeroStartResidualMultiScaleDescriptor
):
    """G16: G15 with a per-location upper bound on residual injection.

    The bound only shrinks an oversized injection. It never enlarges a weak
    residual, and the detached main-path norm prevents the cap itself from
    steering or shrinking the legacy G0 descriptor path.
    """

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        cfg = config or {}
        self.descriptor_injection_norm_control_enabled = bool(
            cfg.get('descriptor_injection_norm_control_enabled', False)
        )
        self.descriptor_injection_ratio_cap = float(
            cfg.get('descriptor_injection_ratio_cap', 0.2)
        )
        if not 0.0 < self.descriptor_injection_ratio_cap <= 1.0:
            raise ValueError(
                'descriptor_injection_ratio_cap must be in (0, 1]'
            )
        self.reset_descriptor_norm_control_epoch_stats()
        print(
            'SuperRetinaWithNormControlledZeroStartMultiScaleDescriptor '
            'initialized: norm control enabled='
            f'{self.descriptor_injection_norm_control_enabled}, '
            f'only-shrink injection cap={self.descriptor_injection_ratio_cap:.4f}'
        )

    def reset_descriptor_norm_control_epoch_stats(self):
        self._descriptor_norm_control_epoch_stats = {
            'calls': 0,
            'raw_injection_ratio': 0.0,
            'controlled_injection_ratio': 0.0,
            'cap_active_fraction': 0.0,
            'scale_mean': 0.0,
        }

    def reset_descriptor_gate_epoch_stats(self):
        super().reset_descriptor_gate_epoch_stats()
        self.reset_descriptor_norm_control_epoch_stats()

    def descriptor_gate_epoch_summary(self):
        summary = super().descriptor_gate_epoch_summary()
        stats = self._descriptor_norm_control_epoch_stats
        calls = max(1, stats['calls'])
        summary.update({
            key: stats[key] / calls
            for key in (
                'raw_injection_ratio',
                'controlled_injection_ratio',
                'cap_active_fraction',
                'scale_mean',
            )
        })
        summary['norm_control_sampled_calls'] = float(stats['calls'])
        return summary

    def _descriptor_injection(
        self, descriptor_main, descriptor_residual
    ):
        gate, agreement = self._descriptor_gate(
            descriptor_main, descriptor_residual
        )
        raw_injection = gate * descriptor_residual
        if not self.descriptor_injection_norm_control_enabled:
            self._record_descriptor_gate_stats(
                gate, descriptor_main, descriptor_residual, agreement
            )
            return raw_injection
        main_norm = torch.norm(descriptor_main, p=2, dim=1, keepdim=True)
        raw_norm = torch.norm(raw_injection, p=2, dim=1, keepdim=True)
        allowed_norm = (
            self.descriptor_injection_ratio_cap * main_norm.detach()
        )
        scale = torch.clamp(
            allowed_norm / raw_norm.clamp_min(1e-6), max=1.0
        )
        controlled_injection = raw_injection * scale

        # Preserve the historical gate statistics for direct G15 comparison.
        self._record_descriptor_gate_stats(
            gate, descriptor_main, descriptor_residual, agreement
        )
        if (
            self.training
            and self.log_descriptor_gate_stats
            and self._descriptor_norm_control_epoch_stats['calls']
                < self.descriptor_gate_stats_max_calls
        ):
            with torch.no_grad():
                controlled_norm = torch.norm(
                    controlled_injection.detach(), p=2, dim=1, keepdim=True
                )
                denominator = main_norm.detach().clamp_min(1e-6)
                stats = self._descriptor_norm_control_epoch_stats
                stats['calls'] += 1
                stats['raw_injection_ratio'] += float(
                    (raw_norm.detach() / denominator).mean().item()
                )
                stats['controlled_injection_ratio'] += float(
                    (controlled_norm / denominator).mean().item()
                )
                stats['cap_active_fraction'] += float(
                    (scale.detach() < 1.0).float().mean().item()
                )
                stats['scale_mean'] += float(scale.detach().mean().item())
        return controlled_injection


class SuperRetinaWithEMATeacherPKE(SuperRetinaWithVesselOnlyMasked):
    """G14: use a frozen EMA copy only for training-time PKE decisions."""

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        cfg = config or {}
        self.pke_ema_teacher_enabled = bool(
            cfg.get('pke_ema_teacher_enabled', False)
        )
        self.pke_ema_teacher_decay = float(
            cfg.get('pke_ema_teacher_decay', 0.99)
        )
        self.pke_ema_teacher_start_epoch = int(
            cfg.get('pke_ema_teacher_start_epoch', 10)
        )
        if not 0.0 <= self.pke_ema_teacher_decay < 1.0:
            raise ValueError('pke_ema_teacher_decay must be in [0, 1)')
        if self.pke_ema_teacher_start_epoch < 0:
            raise ValueError(
                'pke_ema_teacher_start_epoch must be non-negative'
            )
        self.ema_teacher = copy.deepcopy(self)
        self.ema_teacher.pke_ema_teacher_enabled = False
        self.ema_teacher.requires_grad_(False)
        self.ema_teacher.eval()
        self._ema_teacher_updates = 0
        self.to(device)
        print(
            'SuperRetinaWithEMATeacherPKE initialized: '
            f'enabled={self.pke_ema_teacher_enabled}, '
            f'decay={self.pke_ema_teacher_decay:.6f}, '
            f'start_epoch={self.pke_ema_teacher_start_epoch}'
        )

    @torch.no_grad()
    def update_ema_teacher(self):
        """Update after each student optimizer step; teacher never gets grads."""
        decay = (
            0.0
            if self.current_epoch < self.pke_ema_teacher_start_epoch
            else self.pke_ema_teacher_decay
        )
        student_parameters = dict(self.named_parameters())
        for name, teacher_parameter in self.ema_teacher.named_parameters():
            teacher_parameter.mul_(decay).add_(
                student_parameters[name].detach(), alpha=1.0 - decay
            )
        student_buffers = dict(self.named_buffers())
        for name, teacher_buffer in self.ema_teacher.named_buffers():
            teacher_buffer.copy_(student_buffers[name])
        self._ema_teacher_updates += 1

    def ema_teacher_summary(self):
        with torch.no_grad():
            student_parameters = dict(self.named_parameters())
            squared_delta = sum(
                (
                    teacher_parameter
                    - student_parameters[name].detach()
                ).square().sum()
                for name, teacher_parameter
                in self.ema_teacher.named_parameters()
            )
        return {
            'updates': float(self._ema_teacher_updates),
            'active': float(
                self.current_epoch >= self.pke_ema_teacher_start_epoch
            ),
            'decay': float(self.pke_ema_teacher_decay),
            'parameter_delta_l2': float(squared_delta.sqrt().item()),
        }


class SuperRetinaWithMultiScaleDetectorResidual(
    SuperRetinaWithVesselOnlyMasked
):
    """G13: preserve G0 and add a bounded multi-scale detector-logit residual."""

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        cfg = config or {}
        gate_init = float(cfg.get('detector_multiscale_gate_init', 0.05))
        self.detector_multiscale_gate_max = float(
            cfg.get('detector_multiscale_gate_max', 0.15)
        )
        hidden_channels = int(
            cfg.get('detector_multiscale_hidden_channels', 32)
        )
        if not 0 < gate_init < self.detector_multiscale_gate_max <= 1:
            raise ValueError(
                'detector_multiscale_gate_init must be positive and smaller '
                'than detector_multiscale_gate_max <= 1'
            )
        if hidden_channels <= 0:
            raise ValueError(
                'detector_multiscale_hidden_channels must be positive'
            )
        self.detector_residual_conv2 = nn.Sequential(
            nn.Conv2d(64, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.detector_residual_conv3 = nn.Sequential(
            nn.Conv2d(128, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.detector_residual_fusion = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 2, hidden_channels,
                kernel_size=3, padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        # Exact G0 detector at initialization. The fusion output learns first,
        # then gradients begin reaching the shallow/mid-level projections.
        nn.init.zeros_(self.detector_residual_fusion[-1].weight)
        nn.init.zeros_(self.detector_residual_fusion[-1].bias)
        gate_fraction = gate_init / self.detector_multiscale_gate_max
        gate_logit = math.log(gate_fraction / (1.0 - gate_fraction))
        self.detector_multiscale_gate_logit = nn.Parameter(
            torch.tensor(gate_logit, dtype=torch.float32)
        )
        self.log_detector_residual_stats = bool(
            cfg.get('log_detector_residual_stats', False)
        )
        self.detector_residual_stats_max_calls = int(
            cfg.get('detector_residual_stats_max_calls', 64)
        )
        if self.detector_residual_stats_max_calls <= 0:
            raise ValueError(
                'detector_residual_stats_max_calls must be positive'
            )
        self.reset_detector_residual_epoch_stats()
        self._supports_descriptor_only_forward = True
        self.to(device)
        print(
            'SuperRetinaWithMultiScaleDetectorResidual initialized: '
            f'gate_init={gate_init:.4f}, '
            f'gate_max={self.detector_multiscale_gate_max:.4f}, '
            f'hidden_channels={hidden_channels}'
        )

    def reset_detector_residual_epoch_stats(self):
        self._detector_residual_epoch_stats = {
            'calls': 0,
            'gate': 0.0,
            'main_logit_norm': 0.0,
            'residual_logit_norm': 0.0,
            'injection_ratio': 0.0,
        }

    def detector_residual_epoch_summary(self):
        stats = self._detector_residual_epoch_stats
        calls = max(1, stats['calls'])
        return {
            'sampled_calls': float(stats['calls']),
            'gate': stats['gate'] / calls,
            'main_logit_norm': stats['main_logit_norm'] / calls,
            'residual_logit_norm':
                stats['residual_logit_norm'] / calls,
            'injection_ratio': stats['injection_ratio'] / calls,
        }

    def _record_detector_residual_stats(
            self, main_logits, residual_logits, gate):
        if not self.training or not self.log_detector_residual_stats:
            return
        stats = self._detector_residual_epoch_stats
        if stats['calls'] >= self.detector_residual_stats_max_calls:
            return
        with torch.no_grad():
            main_norm = main_logits.detach().flatten(1).norm(dim=1).mean()
            residual_norm = (
                gate.detach() * residual_logits.detach()
            ).flatten(1).norm(dim=1).mean()
            stats['calls'] += 1
            stats['gate'] += float(gate.detach().item())
            stats['main_logit_norm'] += float(main_norm.item())
            stats['residual_logit_norm'] += float(residual_norm.item())
            stats['injection_ratio'] += float(
                (residual_norm / main_norm.clamp_min(1e-6)).item()
            )

    def network(self, x, return_cPa=False, descriptor_only=False):
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)

        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)

        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)

        x = self.relu(self.conv4a(x))
        conv4 = self.relu(self.conv4b(x))

        cDa = self.relu(self.convDa(conv4))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)
        descriptor_norm = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(descriptor_norm, 1))
        desc = self.trans_conv(desc)
        if descriptor_only:
            return desc

        cPa = self.upsample(conv4)
        cPa = torch.cat([cPa, conv3], dim=1)
        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)
        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)
        cPa = self.dconv_up1(cPa)
        main_logits = self.conv_last(cPa)

        residual_conv2 = self.detector_residual_conv2(conv2)
        residual_conv3 = self.detector_residual_conv3(conv3)
        residual_conv3 = F.interpolate(
            residual_conv3,
            size=residual_conv2.shape[-2:],
            mode='bilinear',
            align_corners=True,
        )
        residual_logits = self.detector_residual_fusion(
            torch.cat([residual_conv2, residual_conv3], dim=1)
        )
        residual_logits = F.interpolate(
            residual_logits,
            size=main_logits.shape[-2:],
            mode='bilinear',
            align_corners=True,
        )
        gate = self.detector_multiscale_gate_max * torch.sigmoid(
            self.detector_multiscale_gate_logit
        )
        self._record_detector_residual_stats(
            main_logits, residual_logits, gate
        )
        semi = torch.sigmoid(main_logits + gate * residual_logits)
        if return_cPa:
            return semi, desc, cPa
        return semi, desc


class SuperRetinaWithSpatialGatedMultiScaleDescriptor(
    SuperRetinaWithResidualMultiScaleDescriptor
):
    """G8: G7 with a bounded, learned spatial gate."""

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        cfg = config or {}
        self.descriptor_gate_max = float(
            cfg.get('descriptor_gate_max', 0.2)
        )
        gate_init = float(
            cfg.get('descriptor_multiscale_gate_init', 0.1)
        )
        hidden_channels = int(
            cfg.get('descriptor_spatial_gate_hidden_channels', 32)
        )
        if not 0.0 < gate_init < self.descriptor_gate_max:
            raise ValueError(
                'descriptor_multiscale_gate_init must be between 0 and '
                'descriptor_gate_max'
            )
        if hidden_channels <= 0:
            raise ValueError(
                'descriptor_spatial_gate_hidden_channels must be positive'
            )
        del self.descriptor_multiscale_gate_logit
        self.descriptor_spatial_gate = nn.Sequential(
            nn.Conv2d(512, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        final_layer = self.descriptor_spatial_gate[-1]
        nn.init.zeros_(final_layer.weight)
        initial_probability = gate_init / self.descriptor_gate_max
        nn.init.constant_(
            final_layer.bias,
            math.log(initial_probability / (1.0 - initial_probability)),
        )
        self.to(device)
        print(
            '✅ SuperRetinaWithSpatialGatedMultiScaleDescriptor 初始化完成，'
            f'空间门控上限={self.descriptor_gate_max:.4f}，'
            f'初始门控={gate_init:.4f}，隐藏通道={hidden_channels}'
        )

    def _descriptor_gate(
        self, descriptor_main, descriptor_residual
    ):
        logits = self.descriptor_spatial_gate(torch.cat(
            [descriptor_main, descriptor_residual], dim=1
        ))
        return self.descriptor_gate_max * torch.sigmoid(logits), None


class SuperRetinaWithAgreementGatedMultiScaleDescriptor(
    SuperRetinaWithResidualMultiScaleDescriptor
):
    """G9: G7 with a bounded gate driven by local main/residual agreement."""

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        cfg = config or {}
        self.descriptor_gate_max = float(
            cfg.get('descriptor_gate_max', 0.2)
        )
        center_init = float(
            cfg.get('descriptor_agreement_center_init', 0.0)
        )
        scale_init = float(
            cfg.get('descriptor_agreement_scale_init', 2.0)
        )
        if self.descriptor_gate_max <= 0:
            raise ValueError('descriptor_gate_max must be positive')
        if scale_init <= 0:
            raise ValueError(
                'descriptor_agreement_scale_init must be positive'
            )
        del self.descriptor_multiscale_gate_logit
        self.descriptor_agreement_center = nn.Parameter(
            torch.tensor(center_init, dtype=torch.float32)
        )
        self.descriptor_agreement_scale_raw = nn.Parameter(
            torch.tensor(
                math.log(math.expm1(scale_init)), dtype=torch.float32
            )
        )
        self.to(device)
        print(
            '✅ SuperRetinaWithAgreementGatedMultiScaleDescriptor 初始化完成，'
            f'门控上限={self.descriptor_gate_max:.4f}，'
            f'agreement中心={center_init:.4f}，尺度={scale_init:.4f}'
        )

    def _descriptor_gate(
        self, descriptor_main, descriptor_residual
    ):
        main_centered = descriptor_main.detach() - descriptor_main.detach().mean(
            dim=1, keepdim=True
        )
        residual_centered = descriptor_residual - descriptor_residual.mean(
            dim=1, keepdim=True
        )
        agreement = F.cosine_similarity(
            main_centered, residual_centered, dim=1, eps=1e-6
        ).unsqueeze(1)
        scale = F.softplus(self.descriptor_agreement_scale_raw)
        logits = scale * (
            agreement - self.descriptor_agreement_center
        )
        gate = self.descriptor_gate_max * torch.sigmoid(logits)
        return gate, agreement
