import torch
import scipy.stats as st # 科学计算统计工具，高斯分布计算
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa # 图像增强库，增强器
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from common.common_util import pre_processing
from common.train_util import get_gaussian_kernel
import time

class RetinaDataset(Dataset):
    def __init__(self, data_path, split_file='eccv22_train.txt',
                 is_train=True, data_shape=(768, 768), auxiliary=None):
        self.data = []

        self.image_path = os.path.join(data_path, 'ImageData')
        self.label_path = os.path.join(data_path, 'Annotations')
        self.split_file = os.path.join(data_path, 'ImageSets', split_file)

        # ========== 原模型数据增强（已备份，供将来使用） ==========
        # self.enhancement_sequential_original = iaa.Sequential([
        #     iaa.Multiply((1.0, 1.2)),  # change brightness, doesn't affect keypoints
        #     iaa.Sometimes(
        #         0.2,
        #         iaa.GaussianBlur(sigma=(0, 6))
        #     ),
        #     iaa.Sometimes(
        #         0.2,
        #         iaa.LinearContrast((0.75, 1.2))
        #     ),
        # ], random_order=True)
        
        # ========== 新数据增强（基于数据增强说明.md） ==========
        # 光度变换增强：使用 imgaug 实现
        # 根据文档：在 iaa.Sequential 中，所有变换都会按顺序应用（100%概率），除非使用 iaa.Sometimes() 包装
        # 默认行为：所有在配置中启用的变换都会应用（100%概率）
        # 例外：运动模糊在 max_kernel_size == 3 时使用 iaa.Sometimes(0.5, ...) 包装，应用概率为 50%
        max_kernel_size = 3  # 运动模糊的最大核大小
        aug_list = []
        
        # 1. 随机亮度调整：100%应用（如果配置启用）
        aug_list.append(iaa.Add((-50, 50)))  # 最大绝对变化值50
        
        # 2. 随机对比度调整：100%应用（如果配置启用）
        aug_list.append(iaa.LinearContrast((0.5, 1.5)))  # 对比度强度范围 [0.5, 1.5]
        
        # 3. 高斯噪声：100%应用（如果配置启用）
        aug_list.append(iaa.AdditiveGaussianNoise(scale=(0, 10)))  # 标准差范围 [0, 10]
        
        # 4. 脉冲噪声（Speckle Noise）：100%应用（变换本身总是应用）
        # p=(0, 0.0035) 是像素被替换的概率参数范围，不是变换的应用概率
        aug_list.append(iaa.ImpulseNoise(p=(0, 0.0035)))  # 噪声像素概率范围 [0, 0.0035]
        
        # 5. 运动模糊：根据 max_kernel_size 决定应用概率
        if max_kernel_size > 3:
            # 100%应用，核大小在 [3, max_kernel_size] 范围内随机选择
            aug_list.append(iaa.MotionBlur(k=(3, max_kernel_size)))
        elif max_kernel_size == 3:
            # 50%应用（使用 iaa.Sometimes(0.5, ...) 包装）
            aug_list.append(iaa.Sometimes(0.5, iaa.MotionBlur(k=3)))
        
        self.enhancement_sequential = iaa.Sequential(aug_list, random_order=True)

        self.is_train = is_train
        self.model_image_height, self.model_image_width = data_shape[0], data_shape[1]

        with open(self.split_file) as f:
            files = f.readlines()

        for file in files:
            file = file.strip()
            try:
                image, label = file.split(', ')
                image = os.path.join(self.image_path, image)
                label = os.path.join(self.label_path, label)
                assert os.path.exists(image) and os.path.exists(label)
                self.data.append((image, label))
            except Exception:
                pass
        if auxiliary is not None and os.path.exists(auxiliary):
            print('-'*10+f"Load Lab {'train' if is_train else 'eval'} data and auxiliary data without labels"+'-'*10)
            auxiliaries_tmp = os.listdir(auxiliary)
            auxiliaries = []
            auxiliaries_tmp = [(os.path.join(auxiliary, a), None) for a in auxiliaries_tmp]
            for a, b in auxiliaries_tmp:
                if a.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.ppm')):
                    auxiliaries.append((a, b))
            self.data = self.data + auxiliaries
        else:
            print('-' * 10 + f"Load Lab {'train' if is_train else 'eval'} data, and there is no auxiliary data" + '-' * 10)
        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    # 每一轮取每一张照片的时候会调用这个方法
    def __getitem__(self, index):
        input_with_label = False

        image_path, label_path = self.data[index]
        label_name = '[None]'

        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image)
        image = image[:, :, 1]

        image = pre_processing(image)
        # 数据增强（仅在训练时应用）
        if self.is_train:
            # 将图像转换为uint8格式（0-255）以便应用imgaug增强
            image_uint8 = (image * 255).astype(np.uint8)
            # 应用光度变换增强
            image_uint8 = self.enhancement_sequential(image=image_uint8)
            # 转换回float32格式（0-1）
            image = image_uint8.astype(np.float32) / 255.0
        
        # 转换为PIL Image（需要uint8格式）
        image = Image.fromarray((image * 255).astype(np.uint8))
        image_tensor = self.transforms(image)

        # pytorch要求张量维度（通道数，高，宽）
        # 标准（x，y）对应（宽，高）
        if label_path is not None:
            label_name = os.path.split(label_path)[-1]
            keypoint_position = np.loadtxt(label_path)  # (2, n): (x, y).T
            keypoint_position[:, 0] *= image_tensor.shape[-1]
            keypoint_position[:, 1] *= image_tensor.shape[-2]

            tensor_position = torch.zeros([self.model_image_height, self.model_image_width])

            tensor_position[keypoint_position[:, 1], keypoint_position[:, 0]] = 1
            tensor_position = tensor_position.unsqueeze(0)
            input_with_label = True

            return image_tensor, input_with_label, tensor_position, label_name

        # without labels, only train descriptor
        tensor_position = torch.empty(image_tensor.shape)

        return image_tensor, input_with_label, tensor_position, label_name



