import math
import os
import random
import shutil
import time
import warnings
import torch
from torchvision import transforms as T
from torch.nn import functional as F
import numpy as np
import cv2
import scipy.stats as st
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
from common.inference_diagnostics import write_jsonl


def affine_images(images, used_for='detector'):
    """
    Perform affine transformation on images
    :param images: (B, C, H, W)
    :param keypoint_labels: corresponding labels
    :param value_map: value maps, used to record history learned geo_points
    :return: results of affine images, affine labels, affine value maps, affine transformed grid_inverse, inverse transformed grid_inverse
    """
    h, w = images.shape[2:]
    # 变换矩阵和变换逆矩阵
    theta = None
    thetaI = None
    for i in range(len(images)):
        if used_for == 'detector':
            affine_params = T.RandomAffine(20).get_params(degrees=[-15, 15], translate=[0.2, 0.2],
                                                          scale_ranges=[0.9, 1.35], shears=None, img_size=[h, w])
        else:
            affine_params = T.RandomAffine(20).get_params(degrees=[-3, 3], translate=[0.1, 0.1],
                                                          scale_ranges=[0.9, 1.1], shears=None, img_size=[h, w])
        angle = -affine_params[0] * math.pi / 180
        theta_ = torch.tensor([
            [1 / affine_params[2] * math.cos(angle), math.sin(-angle), -affine_params[1][0] / images.shape[2]],
            [math.sin(angle), 1 / affine_params[2] * math.cos(angle), -affine_params[1][1] / images.shape[3]],
            [0, 0, 1]
        ], dtype=torch.float).to(images)
        thetaI_ = theta_.inverse()
        theta_ = theta_[:2]
        thetaI_ = thetaI_[:2]
        theta_ = theta_.unsqueeze(0)
        thetaI_ = thetaI_.unsqueeze(0)
        if theta is None:
            theta = theta_
        else:
            theta = torch.cat((theta, theta_))
        if thetaI is None:
            thetaI = thetaI_
        else:
            thetaI = torch.cat((thetaI, thetaI_))
    grid = F.affine_grid(theta, images.size(), align_corners=True)
    grid = grid.to(images)
    grid_inverse = F.affine_grid(thetaI, images.size(), align_corners=True)
    grid_inverse = grid_inverse.to(images)
    output = F.grid_sample(images, grid, align_corners=True)
    ### 根据 grid 对图像做双线性插值采样，得到仿射图像
    if used_for == 'descriptor':
        if random.random() >= 0.4:
            output = output.repeat(1, 3, 1, 1)
            output = T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2)(output)
            output = T.Grayscale()(output)
    return output.detach().clone(), grid, grid_inverse


def get_gaussian_kernel(kernlen=21, nsig=5):
    ### nsig是高斯分布标准差
    """Get kernels used for generating Gaussian heatmaps"""
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    ### 采样步长
    kern1d = np.diff(st.norm.cdf(x))
    ###对正态分布的累积分布函数做差分求一维高斯核
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    ### 外积生成二维高斯核，开方平滑
    kernel = kernel_raw / kernel_raw.sum()
    ### 归一化
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    weight = torch.nn.Parameter(data=kernel, requires_grad=False)
    weight = (weight - weight.min()) / (weight.max() - weight.min())
    return weight


def value_map_load(save_dir, names, input_with_label, shape=(768, 768), value_maps_running=None):
    value_maps = []
    for s, name in enumerate(names):
        path = os.path.join(save_dir, name.split('.')[0] + '.png')
        if input_with_label[s] and value_maps_running is not None and name in value_maps_running:
            value_map = value_maps_running[name]
        elif input_with_label[s] and value_maps_running is None and os.path.exists(path):
            value_map = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            value_map = np.zeros([shape[0], shape[1]]).astype(np.uint8)
        value_map = torch.from_numpy(value_map).unsqueeze(0).unsqueeze(0)
        value_maps.append(value_map)
    return torch.cat(value_maps)


def value_map_save(save_dir, names, input_with_label, value_maps, value_maps_running=None):
    for s, name in enumerate(names):
        if input_with_label[s]:
            vp = value_maps[s].squeeze().numpy()
            if value_maps_running is not None:
                value_maps_running[name] = vp
            else:
                path = os.path.join(save_dir, name.split('.')[0] + '.png')
                cv2.imwrite(path, vp)


def conflict_projected_backward(model, detector_loss, descriptor_loss):
    """Apply symmetric PCGrad to shared encoder parameters and set ``.grad``."""
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    detector_gradients = torch.autograd.grad(
        detector_loss, parameters, retain_graph=True, allow_unused=True
    )
    descriptor_gradients = torch.autograd.grad(
        descriptor_loss, parameters, allow_unused=True
    )
    shared_prefixes = (
        'conv1a.', 'conv1b.', 'conv2a.', 'conv2b.',
        'conv3a.', 'conv3b.', 'conv4a.', 'conv4b.',
        'network.conv1a.', 'network.conv1b.',
        'network.conv2a.', 'network.conv2b.',
        'network.conv3a.', 'network.conv3b.',
        'network.conv4a.', 'network.conv4b.',
    )
    shared_indices = [
        index for index, (name, _) in enumerate(named_parameters)
        if name.startswith(shared_prefixes)
    ]
    shared_pairs = [
        (detector_gradients[index], descriptor_gradients[index])
        for index in shared_indices
        if detector_gradients[index] is not None
        and descriptor_gradients[index] is not None
    ]
    dot = sum(
        (detector_gradient * descriptor_gradient).sum()
        for detector_gradient, descriptor_gradient in shared_pairs
    ) if shared_pairs else detector_loss.new_tensor(0.0)
    detector_norm_sq = sum(
        detector_gradient.square().sum()
        for detector_gradient, _ in shared_pairs
    ) if shared_pairs else detector_loss.new_tensor(0.0)
    descriptor_norm_sq = sum(
        descriptor_gradient.square().sum()
        for _, descriptor_gradient in shared_pairs
    ) if shared_pairs else detector_loss.new_tensor(0.0)
    denominator = (
        detector_norm_sq.sqrt() * descriptor_norm_sq.sqrt()
    ).clamp_min(1e-12)
    cosine_before = dot / denominator
    conflict = bool(dot.detach().item() < 0)
    projected_detector = {}
    projected_descriptor = {}
    if conflict and shared_pairs:
        for index in shared_indices:
            detector_gradient = detector_gradients[index]
            descriptor_gradient = descriptor_gradients[index]
            if detector_gradient is None or descriptor_gradient is None:
                continue
            projected_detector[index] = (
                detector_gradient
                - dot / descriptor_norm_sq.clamp_min(1e-12)
                * descriptor_gradient
            )
            projected_descriptor[index] = (
                descriptor_gradient
                - dot / detector_norm_sq.clamp_min(1e-12)
                * detector_gradient
            )

    projected_pairs = []
    removed_norm_sq = detector_loss.new_tensor(0.0)
    original_norm_sq = detector_loss.new_tensor(0.0)
    for index, (_, parameter) in enumerate(named_parameters):
        detector_gradient = detector_gradients[index]
        descriptor_gradient = descriptor_gradients[index]
        if index in projected_detector:
            detector_projected = projected_detector[index]
            descriptor_projected = projected_descriptor[index]
            combined = detector_projected + descriptor_projected
            projected_pairs.append(
                (detector_projected, descriptor_projected)
            )
            removed_norm_sq = removed_norm_sq + (
                detector_projected - detector_gradient
            ).square().sum() + (
                descriptor_projected - descriptor_gradient
            ).square().sum()
            original_norm_sq = original_norm_sq + (
                detector_gradient.square().sum()
                + descriptor_gradient.square().sum()
            )
        elif detector_gradient is None:
            combined = descriptor_gradient
        elif descriptor_gradient is None:
            combined = detector_gradient
        else:
            combined = detector_gradient + descriptor_gradient
        parameter.grad = None if combined is None else combined.detach()

    if projected_pairs:
        projected_dot = sum(
            (detector_gradient * descriptor_gradient).sum()
            for detector_gradient, descriptor_gradient in projected_pairs
        )
        projected_detector_norm = sum(
            detector_gradient.square().sum()
            for detector_gradient, _ in projected_pairs
        ).sqrt()
        projected_descriptor_norm = sum(
            descriptor_gradient.square().sum()
            for _, descriptor_gradient in projected_pairs
        ).sqrt()
        cosine_after = projected_dot / (
            projected_detector_norm * projected_descriptor_norm
        ).clamp_min(1e-12)
    else:
        cosine_after = cosine_before
    removed_fraction = (
        removed_norm_sq.sqrt() / original_norm_sq.sqrt().clamp_min(1e-12)
        if conflict else detector_loss.new_tensor(0.0)
    )
    return {
        'conflict': float(conflict),
        'cosine_before': float(cosine_before.detach().item()),
        'cosine_after': float(cosine_after.detach().item()),
        'detector_grad_norm': float(detector_norm_sq.sqrt().detach().item()),
        'descriptor_grad_norm': float(
            descriptor_norm_sq.sqrt().detach().item()
        ),
        'removed_fraction': float(removed_fraction.detach().item()),
    }


def train_model(model, optimizer, dataloaders, device, num_epochs, train_config, start_epoch=0):
    model_save_path = train_config['model_save_path']
    model_save_epoch = train_config['model_save_epoch']
    pke_start_epoch = train_config['pke_start_epoch']
    pke_show_epoch = train_config['pke_show_epoch']
    pke_show_list = train_config['pke_show_list']
    is_value_map_save = train_config['is_value_map_save']
    value_map_save_dir = train_config['value_map_save_dir']
    resume_value_map = train_config.get('resume_value_map', False)
    shared_gradient_mode = train_config.get(
        'shared_gradient_mode', 'standard'
    )
    if shared_gradient_mode not in {'standard', 'pcgrad'}:
        raise ValueError(
            f'Unknown shared_gradient_mode: {shared_gradient_mode}'
        )
    model._capture_gradient_audit_losses = (
        shared_gradient_mode == 'pcgrad'
    )
    extra_save_epochs = set(train_config.get('extra_save_epochs', []))
    checkpoint_save_epochs_raw = train_config.get('checkpoint_save_epochs')
    checkpoint_path_template = train_config.get('checkpoint_path_template')
    checkpoint_save_epochs = (
        None if checkpoint_save_epochs_raw is None
        else set(int(epoch) for epoch in checkpoint_save_epochs_raw)
    )
    if (checkpoint_save_epochs is None) != (checkpoint_path_template is None):
        raise ValueError(
            'checkpoint_save_epochs and checkpoint_path_template must be configured together'
        )
    if checkpoint_save_epochs is not None:
        final_epoch = num_epochs - 1
        if final_epoch not in checkpoint_save_epochs:
            raise ValueError('checkpoint_save_epochs must include the final epoch')
        expected_final_path = checkpoint_path_template.format(epoch=final_epoch)
        if os.path.abspath(expected_final_path) != os.path.abspath(model_save_path):
            raise ValueError(
                'model_save_path must equal checkpoint_path_template at the final epoch'
            )
        if len({
            checkpoint_path_template.format(epoch=epoch)
            for epoch in checkpoint_save_epochs
        }) != len(checkpoint_save_epochs):
            raise ValueError('checkpoint_path_template must produce a unique path per epoch')
    save_pke_diagnostics = bool(train_config.get('save_pke_diagnostics', False))
    pke_diagnostics_path = train_config.get(
        'pke_diagnostics_path',
        os.path.join(os.path.dirname(model_save_path), 'pke_diagnostics.jsonl'),
    )
    if save_pke_diagnostics and os.path.exists(pke_diagnostics_path):
        os.remove(pke_diagnostics_path)

    if start_epoch < 0:
        raise ValueError('start_epoch must be >= 0')
    if start_epoch >= num_epochs:
        raise ValueError(f'start_epoch ({start_epoch}) must be smaller than num_epochs ({num_epochs})')
    if resume_value_map and not is_value_map_save:
        raise ValueError('resume_value_map=True requires is_value_map_save=True; in-memory value_map cannot be resumed across runs')

    if (
        train_config.get('refuse_existing_experiment_outputs', False)
        and start_epoch == 0
    ):
        checkpoint_paths = (
            [model_save_path] if checkpoint_save_epochs is None
            else [
                checkpoint_path_template.format(epoch=epoch)
                for epoch in checkpoint_save_epochs
            ]
        )
        occupied = [path for path in checkpoint_paths if os.path.exists(path)]
        if os.path.isdir(value_map_save_dir) and os.listdir(value_map_save_dir):
            occupied.append(value_map_save_dir)
        if occupied:
            raise FileExistsError(
                'Refusing to overwrite existing experiment output(s): '
                + ', '.join(occupied)
            )

    model_save_dir = os.path.dirname(model_save_path)
    if model_save_dir:
        os.makedirs(model_save_dir, exist_ok=True)

    pretrained_path = train_config.get('pretrained_path')
    if pretrained_path and os.path.abspath(pretrained_path) == os.path.abspath(model_save_path):
        raise ValueError('model_save_path must be different from pretrained_path to avoid overwriting the source checkpoint')

    if is_value_map_save:
        if os.path.exists(value_map_save_dir):
            if not resume_value_map:
                shutil.rmtree(value_map_save_dir)
                os.makedirs(value_map_save_dir)
        else:
            if resume_value_map and start_epoch > 0:
                raise FileNotFoundError(
                    f'resume_value_map=True but value_map_save_dir does not exist: {value_map_save_dir}'
                )
            os.makedirs(value_map_save_dir)
    elif start_epoch > 0:
        warnings.warn(
            'Resuming from a checkpoint without persisted value_map. '
            'Model/optimizer/epoch will resume, but PKE sample history will be reinitialized.',
            RuntimeWarning,
        )
    
    value_maps_running = None
    if not is_value_map_save:
        value_maps_running = {}

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        model.current_epoch = epoch
        if hasattr(model, 'reset_descriptor_gate_epoch_stats'):
            model.reset_descriptor_gate_epoch_stats()
        if hasattr(model, 'reset_dense_descriptor_epoch_stats'):
            model.reset_dense_descriptor_epoch_stats()
        if hasattr(model, 'reset_detector_residual_epoch_stats'):
            model.reset_detector_residual_epoch_stats()
        # Each epoch may have some phases
        phases = list(dataloaders.keys())
        model.PKE_learn = True
        if epoch < pke_start_epoch:
            model.PKE_learn = False

        for phase in phases:
            pke_show = train_config['pke_show']
            image_shows = []
            init_kp_shows = []
            label_shows = []
            enhanced_kp_shows = []
            show_names = []
            if 'val' in phase:
                if checkpoint_save_epochs is not None:
                    should_save = epoch in checkpoint_save_epochs
                else:
                    should_save = (
                        epoch % model_save_epoch == 0
                        or epoch in extra_save_epochs
                    )
                if should_save:
                    if checkpoint_save_epochs is not None:
                        save_path = checkpoint_path_template.format(epoch=epoch)
                    elif epoch in extra_save_epochs:
                        base_dir = os.path.dirname(model_save_path)
                        base_name = os.path.splitext(os.path.basename(model_save_path))[0]
                        ext = os.path.splitext(model_save_path)[1] or '.pth'
                        save_path = os.path.join(base_dir, f'{base_name}_epoch{epoch}{ext}')
                    else:
                        save_path = model_save_path
                    print(f'save model for epoch {epoch} -> {save_path}')
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(state, save_path)
                continue
            print('-' * 10 + 'phase:' + phase + '\t PKE_learn:' + str(model.PKE_learn) + '-' * 10)
            if 'train' in phase:
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            epoch_samples = 0
            print_descriptor_loss = 0
            print_detector_loss = 0
            num_learned_pts = 0
            num_input_with_label = 0
            num_input_descriptor = 0
            descriptor_supervision_batches = 0
            descriptor_supervision_skipped_batches = 0
            descriptor_supervision_over_limit_images = 0
            descriptor_supervision_total_correspondences = 0
            descriptor_supervision_used_correspondences = 0
            pcgrad_stats = {
                'batches': 0,
                'conflicts': 0.0,
                'cosine_before': 0.0,
                'cosine_after': 0.0,
                'detector_grad_norm': 0.0,
                'descriptor_grad_norm': 0.0,
                'removed_fraction': 0.0,
            }

            for images, input_with_label, keypoint_positions, label_names \
                    in tqdm(dataloaders[phase]):
                batch_size = images.shape[0]
                learn_index = torch.where(input_with_label)
                images = images.to(device)
                value_maps = value_map_load(value_map_save_dir, label_names, input_with_label, images.shape[-2:],
                                            value_maps_running)
                value_maps = value_maps.to(device)
                keypoint_positions = keypoint_positions.to(device)
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled('train' in phase):
                    loss, number_pts_one, print_loss_detector_one, print_loss_descriptor_one, enhanced_label_pts, \
                    enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num \
                    = model(images, keypoint_positions, value_maps, learn_index)
                    descriptor_stats = getattr(
                        model, '_descriptor_supervision_audit', None
                    )
                    if (
                        train_config.get('log_descriptor_supervision_stats', False)
                        and descriptor_stats is not None
                    ):
                        descriptor_supervision_batches += 1
                        counts = descriptor_stats['sample_counts']
                        participating = descriptor_stats['participating_indices']
                        descriptor_supervision_total_correspondences += sum(counts)
                        descriptor_supervision_used_correspondences += sum(
                            counts[index] for index in participating
                        )
                        descriptor_supervision_over_limit_images += len(
                            descriptor_stats['over_limit_indices']
                        )
                        descriptor_supervision_skipped_batches += int(
                            descriptor_stats['exit_reason'] != 'trained'
                        )
                    if save_pke_diagnostics and getattr(model, 'last_pke_diagnostics', None) is not None:
                        for local_index, diagnostic in enumerate(model.last_pke_diagnostics):
                            batch_index = int(learn_index[0][local_index])
                            write_jsonl(pke_diagnostics_path, {
                                'epoch': int(epoch),
                                'phase': phase,
                                'image_name': str(label_names[batch_index]),
                                **diagnostic,
                            })
                    if enhanced_label_pts is None:
                        enhanced_label_pts = keypoint_positions

                    if pke_show:
                        if len(pke_show_list) == 0 and len(learn_index[0]) != 0:
                            # randomly show one image
                            show_names = [(label_names[learn_index[0][0]], learn_index[0][0])]
                            pke_show = False
                        for s, label_path in enumerate(label_names):
                            file_name = os.path.split(label_path)[-1].split('.')[0]
                            if file_name in pke_show_list:
                                show_names.append((label_path, s))
                        for (show_name, idx) in show_names:
                            image_shows.append(images[idx][0].cpu().data)
                            init_kp_shows.append(keypoint_positions[idx][0].cpu().data)
                            label_shows.append(enhanced_label[idx][0].cpu().data)
                            pred_show = detector_pred.detach().cpu()[idx][0]
                            enhanced_kp_shows.append(enhanced_label_pts[idx][0].cpu().data)

                print_detector_loss += print_loss_detector_one * len(learn_index[0])
                ### 累积检测损失
                print_descriptor_loss += print_loss_descriptor_one * batch_size
                num_input_with_label += loss_detector_num
                ### 累计有标签样本数
                num_learned_pts += number_pts_one

                if 'train' in phase:
                    if shared_gradient_mode == 'pcgrad':
                        component_losses = getattr(
                            model, '_gradient_audit_losses', None
                        )
                        if component_losses is None:
                            raise RuntimeError(
                                'PCGrad requires live detector and descriptor losses'
                            )
                        batch_pcgrad = conflict_projected_backward(
                            model, *component_losses
                        )
                        pcgrad_stats['batches'] += 1
                        pcgrad_stats['conflicts'] += batch_pcgrad['conflict']
                        for key in (
                            'cosine_before', 'cosine_after',
                            'detector_grad_norm', 'descriptor_grad_norm',
                            'removed_fraction',
                        ):
                            pcgrad_stats[key] += batch_pcgrad[key]
                    else:
                        loss.backward()
                    optimizer.step()
                    # 定期清理显存，避免碎片化
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # update value map
                if len(learn_index[0]) != 0:
                    value_maps = value_maps.cpu()
                    value_map_save(value_map_save_dir, label_names, input_with_label, value_maps, value_maps_running)

                # statistics
                num_input_descriptor += loss_descriptor_num
                epoch_samples += batch_size

                # 释放不再需要的变量
                del images, value_maps, keypoint_positions
                if 'train' in phase:
                    del loss, detector_pred

            ###################################################################################
            print_detector_loss = print_detector_loss / num_input_with_label
            print_descriptor_loss = print_descriptor_loss / epoch_samples
            print(phase, 'overall loss: {}'.format(print_detector_loss + print_descriptor_loss),
                  'detector_loss: {} of {} nums, #avg learned keypoints:{} '.format(print_detector_loss,
                                                                                    num_input_with_label,
                                                                                    num_learned_pts / num_input_with_label),
                  'descriptor_loss: {} of {} nums'.format(print_descriptor_loss, num_input_descriptor))
            if (
                train_config.get('log_descriptor_supervision_stats', False)
                and descriptor_supervision_batches > 0
            ):
                effective = (
                    descriptor_supervision_used_correspondences
                    / max(1, descriptor_supervision_total_correspondences)
                )
                print(
                    'descriptor supervision: '
                    f'batches={descriptor_supervision_batches}, '
                    f'skipped_batches={descriptor_supervision_skipped_batches}, '
                    f'over_limit_images={descriptor_supervision_over_limit_images}, '
                    f'total_correspondences={descriptor_supervision_total_correspondences}, '
                    f'used_correspondences={descriptor_supervision_used_correspondences}, '
                    f'effective_fraction={effective:.6f}'
                )
            if (
                phase == 'train'
                and train_config.get('log_descriptor_gate_stats', False)
                and hasattr(model, 'descriptor_gate_epoch_summary')
            ):
                gate_stats = model.descriptor_gate_epoch_summary()
                print(
                    'descriptor gate: '
                    + ', '.join(
                        f'{key}={value:.6f}'
                        for key, value in gate_stats.items()
                    )
                )
            if phase == 'train' and pcgrad_stats['batches']:
                pcgrad_batches = pcgrad_stats['batches']
                print(
                    'pcgrad: '
                    f'batches={pcgrad_batches}, '
                    f'conflict_fraction='
                    f"{pcgrad_stats['conflicts'] / pcgrad_batches:.6f}, "
                    f"cosine_before={pcgrad_stats['cosine_before'] / pcgrad_batches:.6f}, "
                    f"cosine_after={pcgrad_stats['cosine_after'] / pcgrad_batches:.6f}, "
                    f"detector_grad_norm={pcgrad_stats['detector_grad_norm'] / pcgrad_batches:.6f}, "
                    f"descriptor_grad_norm={pcgrad_stats['descriptor_grad_norm'] / pcgrad_batches:.6f}, "
                    f"removed_fraction={pcgrad_stats['removed_fraction'] / pcgrad_batches:.6f}"
                )
            if (
                phase == 'train'
                and train_config.get('log_dense_descriptor_stats', False)
                and hasattr(model, 'dense_descriptor_epoch_summary')
            ):
                dense_stats = model.dense_descriptor_epoch_summary()
                print(
                    'balanced dense descriptor: '
                    + ', '.join(
                        f'{key}={value:.6f}'
                        for key, value in dense_stats.items()
                    )
                )
            if (
                phase == 'train'
                and train_config.get('log_detector_residual_stats', False)
                and hasattr(model, 'detector_residual_epoch_summary')
            ):
                detector_stats = model.detector_residual_epoch_summary()
                print(
                    'detector residual: '
                    + ', '.join(
                        f'{key}={value:.6f}'
                        for key, value in detector_stats.items()
                    )
                )

            for s, (name, _) in enumerate(show_names):
                if not epoch % pke_show_epoch == 0:
                    break
                input_show = image_shows[s]
                label_show = label_shows[s]
                init_kp_show = init_kp_shows[s]
                enhanced_kp_show = enhanced_kp_shows[s]
                name = os.path.split(name)[-1].split('.')[0]
                plt.figure(dpi=100)
                plt.imshow(input_show, 'gray')
                plt.title('epoch:{}, phase:{}, name:{}'.format(epoch, phase, name))
                plt.axis('off')
                try:
                    y, x = torch.where(enhanced_kp_show.cpu() == 1)
                    plt.scatter(x, y, s=2, c='springgreen')
                except Exception:
                    pass
                try:
                    y, x = torch.where(init_kp_show.cpu() == 1)
                    plt.scatter(x, y, s=2, c='b')
                except Exception:
                    pass
                # plt.savefig(f'./data/draw/train_new_new/{name1}_epo{epoch}_{len(x)}.png',bbox_inches='tight',pad_inches = -0.1)
                plt.show()
                plt.figure(dpi=200)
                plt.subplot(121)
                plt.imshow(pred_show, 'gray')
                plt.subplot(122)
                plt.imshow(label_show, 'gray')
                plt.show()
                plt.close()
                plt.pause(0.1)
