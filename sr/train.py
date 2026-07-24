import argparse
import torch
import os
import random

from torch.utils.data import DataLoader

from common.train_util import train_model
from dataset.retina_dataset import RetinaDataset
from model.super_retina import (
    SuperRetina,
    SuperRetinaFPN,
    SuperRetinaWithPerceptualLoss,
    SuperRetinaWithVesselRegularization,
    SuperRetinaWithVesselOnly,
    SuperRetinaWithVesselOnlyMasked,
)
import torch.optim as optim
import yaml
import warnings
import numpy as np


def set_global_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import imgaug as ia
        ia.seed(seed)
    except Exception:
        pass

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import imgaug as ia
        ia.seed(worker_seed)
    except Exception:
        pass

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # 设置 CUDA 线性代数后端，避免 cusolver 错误
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.preferred_linalg_library('magma')
        except:
            # 如果 magma 不可用，尝试其他后端
            try:
                torch.backends.cuda.preferred_linalg_library('cusolver')
            except:
                pass

    parser = argparse.ArgumentParser(description='Train SuperRetina variants')
    parser.add_argument(
        '--config',
        type=str,
        default='/home/data1/zhangjunhong/sr_project/sr/config/train.yaml',
        help='Path to train yaml (default: config/train.yaml)',
    )
    args = parser.parse_args()
    config_path = args.config

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError("Config File doesn't Exist")

    assert 'MODEL' in config
    assert 'PKE' in config
    assert 'DATASET' in config
    assert 'VALUE_MAP' in config
    train_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}

    batch_size = train_config['batch_size']
    num_epoch = train_config['num_epoch']
    seed = int(train_config.get('seed', 3407))
    deterministic = bool(train_config.get('deterministic', True))
    num_workers = int(train_config.get('num_workers', 8))
    set_global_seed(seed, deterministic=deterministic)

    device = train_config['device']
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    dataset_path = train_config['dataset_path']
    data_shape = (train_config['model_image_height'], train_config['model_image_width'])

    train_split_file = train_config['train_split_file']
    val_split_file = train_config['val_split_file']
    auxiliary = train_config['auxiliary']
    train_set = RetinaDataset(dataset_path, split_file=train_split_file,
                            is_train=True, data_shape=data_shape, auxiliary=auxiliary)
    val_set = RetinaDataset(dataset_path, split_file=val_split_file, is_train=False, data_shape=data_shape)

    load_pre_trained_model = train_config['load_pre_trained_model']
    pretrained_path = train_config['pretrained_path']
    model_variant = train_config.get('model_variant', 'baseline')
    resume_optimizer = train_config.get('resume_optimizer', True)

    if model_variant == 'baseline':
        model = SuperRetina(train_config, device=device)
    elif model_variant == 'fpn':
        model = SuperRetinaFPN(train_config, device=device)
    elif model_variant == 'perceptual':
        model = SuperRetinaWithPerceptualLoss(train_config, device=device)
    elif model_variant == 'vessel_regularization':
        model = SuperRetinaWithVesselRegularization(train_config, device=device)
    elif model_variant == 'vessel_only':
        model = SuperRetinaWithVesselOnly(train_config, device=device)
    elif model_variant == 'vessel_only_masked':
        model = SuperRetinaWithVesselOnlyMasked(train_config, device=device)
    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0

    if load_pre_trained_model:
        if not os.path.exists(pretrained_path):
            raise Exception('Pretrained model doesn\'t exist')
        checkpoint = torch.load(pretrained_path, map_location=device)
        if hasattr(model, 'load_pretrained_weights'):
            model.load_pretrained_weights(pretrained_path, device=device, strict=False)
        else:
            model.load_state_dict(checkpoint['net'])
        if resume_optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch']) + 1

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)

    dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                worker_init_fn=seed_worker, generator=train_generator),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              worker_init_fn=seed_worker, generator=val_generator)
        }

    print("🔍 GPU 检查报告:")
    print(f"  - torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  - 当前 device: {device}")
    print(f"  - 模型是否在 GPU 上: {next(model.parameters()).device}")  # 关键一行
    print(f"  - batch_size: {batch_size}, num_workers: 查看 DataLoader")
    print(f"  - seed: {seed}, deterministic: {deterministic}, num_workers: {num_workers}")
    print(f"  - model_variant: {model_variant}")
    print(f"  - start_epoch: {start_epoch}")
    print(f"  - resume_optimizer: {resume_optimizer}")
    print(f"  - is_value_map_save: {train_config.get('is_value_map_save')}")
    print(f"  - resume_value_map: {train_config.get('resume_value_map', False)}")
    print(f"  - pke_content_mode: {train_config.get('pke_content_mode', 'one_way')}")
    print(f"  - pke_content_weak_feedback: {train_config.get('pke_content_weak_feedback', False)}")
    print(f"  - pke_content_strong_feedback_multiplier: "
          f"{train_config.get('pke_content_strong_feedback_multiplier', 1)}")
    print(f"  - pke_content_weak_feedback_multiplier: "
          f"{train_config.get('pke_content_weak_feedback_multiplier', 1)}")
    print(f"  - pke_content_mode: {train_config.get('pke_content_mode', 'one_way')}")
    print(f"  - pke_content_weak_feedback: {train_config.get('pke_content_weak_feedback', False)}")
    print(f"  - pke_content_strong_feedback_multiplier: "
          f"{train_config.get('pke_content_strong_feedback_multiplier', 1)}")
    print(f"  - pke_content_weak_feedback_multiplier: "
          f"{train_config.get('pke_content_weak_feedback_multiplier', 1)}")

    model = train_model(
        model,
        optimizer,
        dataloaders,
        device,
        num_epochs=num_epoch,
        train_config=train_config,
        start_epoch=start_epoch,
    )




