import random
import sys
import time

from model.pke_module import pke_learn

from torch.nn import functional as F
import torch
import torch.nn as nn

from loss.dice_loss import DiceBCELoss, DiceLoss
from loss.triplet_loss import triplet_margin_loss_gor, triplet_margin_loss_gor_one, sos_reg
from loss.perceptual_loss import PerceptualLoss

from common.common_util import remove_borders, sample_keypoint_desc, simple_nms, nms, \
    sample_descriptors
from common.train_util import get_gaussian_kernel, affine_images

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
                               affine_detector_pred=affine_detector_pred)

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
            if n > 1000:  # avoid OOM
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
                dis = torch.norm(descriptor - affine_descriptor, dim=0)
                dis[ar, ar] = dis.max() + 1
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
    """

    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__(config=config, device=device, n_class=n_class)
        self.perceptual_loss = PerceptualLoss(device=device)
        self.perceptual_weight = config.get('perceptual_weight', 0.05) if config is not None else 0.05
        print(f"✅ SuperRetinaWithPerceptualLoss 初始化完成，perceptual_weight={self.perceptual_weight}（单层 relu4_2）")

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

            # === 0.05 单层感知损失（核心恢复点）===
            if self.PKE_learn and len(learn_index[0]) != 0 and hasattr(self, 'perceptual_loss'):
                perc_input = affine_detector_pred.repeat(1, 3, 1, 1)
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
                loss_descriptor_num = loss_descriptor_num

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

            # === 原有 0.05 perceptual loss（完全不变）===
            if self.PKE_learn and len(learn_index[0]) != 0 and hasattr(self, 'perceptual_loss'):
                perc_input = affine_detector_pred.repeat(1, 3, 1, 1)
                perc_target = detector_pred[learn_index].repeat(1, 3, 1, 1)
                perc_loss = self.perceptual_loss(perc_input, perc_target)
                loss_detector = loss_detector + self.perceptual_weight * perc_loss

            # === 新增：vessel regularization（使用 enhanced_label 作为 pseudo vessel mask）===
            if cPa is not None and enhanced_label is not None:
                vessel_pred = self.vessel_head(cPa)                    # vessel head 使用 decoder 特征
                vessel_loss = loss_cal(vessel_pred, enhanced_label)    # DiceLoss
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
                loss_descriptor_num = loss_descriptor_num

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
            if cPa is not None and enhanced_label is not None:
                vessel_pred = self.vessel_head(cPa)
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
