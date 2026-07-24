import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from common.common_util import sample_keypoint_desc, nms
from model.record_module import update_value_map


def mapping_points(grid, points, h, w):
    """ Using grid_inverse to apply affine transform on geo_points
        :return point set and its corresponding affine point set
    """

    grid_points = [(grid[s, k[:, 1].long(), k[:, 0].long()]) for s, k in
                   enumerate(points)]
    filter_points = []
    affine_points = []
    for s, k in enumerate(grid_points):  # filter bad geo_points
        idx = (k[:, 0] < 1) & (k[:, 0] > -1) & (k[:, 1] < 1) & (
                k[:, 1] > -1)
        gp = grid_points[s][idx]
        gp[:, 0] = (gp[:, 0] + 1) / 2 * (w - 1)
        gp[:, 1] = (gp[:, 1] + 1) / 2 * (h - 1)
        affine_points.append(gp)
        filter_points.append(points[s][idx])

    return filter_points, affine_points


def content_filter(descriptor_pred, affine_descriptor_pred, geo_points,
                   affine_geo_points, content_thresh=0.7, scale=8,
                   mode='one_way', weak_feedback=False,
                   strong_feedback_multiplier=1, weak_feedback_multiplier=1,
                   return_feedback_weights=False):
    """
    content-based matching in paper
    :param descriptor_pred: descriptors of input_image images
    :param affine_descriptor_pred: descriptors of affine images
    :param geo_points: 
    :param affine_geo_points:
    :param content_thresh:
    :param scale: down sampling size of descriptor_pred
    :param mode: ``one_way`` preserves the original PKE criterion.  In
        ``bidirectional`` mode the known affine correspondence must also be
        the nearest-neighbour ratio match in the reverse direction.
    :param weak_feedback: in bidirectional mode, retain forward-valid but
        reverse-invalid points with a smaller value-map update multiplier.
        They remain subject to the original geometry and forward content gate.
    :return: content-filtered keypoints, and optionally integer feedback weights
    """

    descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                   for k, d in zip(geo_points, descriptor_pred)]
    aff_descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                       for k, d in zip(affine_geo_points, affine_descriptor_pred)]
    content_points = []
    affine_content_points = []
    feedback_weights = []
    if mode not in {'one_way', 'bidirectional'}:
        raise ValueError(f'Unknown PKE content mode: {mode}')
    if strong_feedback_multiplier < 1 or weak_feedback_multiplier < 1:
        raise ValueError('PKE feedback multipliers must be at least 1')
    dist = [torch.norm(descriptors[d][:, None] - aff_descriptors[d], dim=2, p=2)
            for d in range(len(descriptors))]
    for i in range(len(dist)):
        D = dist[i]
        if len(D) <= 1:
            content_points.append([])
            affine_content_points.append([])
            feedback_weights.append(torch.empty(0, dtype=torch.long, device=D.device))
            continue
        val, ind = torch.topk(D, 2, dim=1, largest=False)

        arange = torch.arange(len(D))
        # rule1 spatial correspondence
        c1 = ind[:, 0] == arange.to(ind.device)
        # rule2 pass the ratio test
        c2 = val[:, 0] < val[:, 1] * content_thresh

        forward_check = c2 & c1
        if mode == 'one_way':
            strong_check = forward_check
            weak_check = torch.zeros_like(forward_check)
        else:
            reverse_val, reverse_ind = torch.topk(D, 2, dim=0, largest=False)
            reverse_check = (
                (reverse_ind[0] == arange.to(reverse_ind.device))
                & (reverse_val[0] < reverse_val[1] * content_thresh)
            )
            strong_check = forward_check & reverse_check
            weak_check = forward_check & ~reverse_check if weak_feedback else torch.zeros_like(forward_check)
        check = strong_check | weak_check
        content_points.append(geo_points[i][check])
        affine_content_points.append(affine_geo_points[i][check])
        weights = torch.full(
            (len(D),), strong_feedback_multiplier, dtype=torch.long, device=D.device
        )
        if weak_feedback:
            weights[weak_check] = weak_feedback_multiplier
        feedback_weights.append(weights[check])
    if return_feedback_weights:
        return content_points, affine_content_points, feedback_weights
    return content_points, affine_content_points


def geometric_filter(affine_detector_pred, points, affine_points, max_num=1024, geometric_thresh=0.5,
                     vessel_masks=None, relaxed_non_core_thresh=None):
    """
    geometric matching in paper
    :param affine_detector_pred: geo_points probability of affine image
    :param points: nms results of input_image image
    :param affine_points: nms results of affine image
    :param max_num: maximum number of learned keypoints
    :param geometric_thresh: 
    :return: geometric-filtered keypoints
    """
    geo_points = []
    affine_geo_points = []
    for s, k in enumerate(affine_points):
        sample_aff_values = affine_detector_pred[s, 0, k[:, 1].long(), k[:, 0].long()]
        check = sample_aff_values.squeeze() >= geometric_thresh
        if vessel_masks is not None and relaxed_non_core_thresh is not None:
            # Preserve the strict threshold for vessel core.  Border/non-vessel
            # candidates in the gray interval are admitted to the existing
            # descriptor content filter; no point is fabricated or accepted
            # without that later verification.
            vessel = vessel_masks[s:s + 1].float()
            # Match the 3x3 elliptical (cross-shaped at this size) erosion
            # used by the D2 audit to distinguish vessel core from edge.
            cross_kernel = vessel.new_tensor(
                [[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]
            ).view(1, 1, 3, 3)
            core = F.conv2d(vessel, cross_kernel, padding=1) >= 5.0
            point_is_core = core[0, 0, points[s][:, 1].long(), points[s][:, 0].long()]
            relaxed = sample_aff_values.squeeze() >= relaxed_non_core_thresh
            check = check | (relaxed & ~point_is_core)
        geo_points.append(points[s][check][:max_num])
        affine_geo_points.append(k[check][:max_num])

    return geo_points, affine_geo_points


def pke_learn(detector_pred, descriptor_pred, grid_inverse, affine_detector_pred,
              affine_descriptor_pred, kernel, loss_cal, label_point_positions,
              value_map, config, PKE_learn=True, return_stage_points=False,
              vessel_masks=None, relaxed_non_core_thresh=None):
    """
    pke process used for detector
    :param detector_pred: probability map from raw image
    :param descriptor_pred: prediction of descriptor_pred network
    :param kernel: used for gaussian heatmaps
    :param mask_kernel: used for masking initial keypoints
    :param grid_inverse: used for inverse
    :param loss_cal: loss (default is dice)
    :param label_point_positions: positions of keypoints on labels
    :param value_map: value map for recoding and selecting learned geo_points
    :param pke_learn: whether to use PKE
    :return: loss of detector, num of additional geo_points, updated value maps and enhanced labels
    """
    # used for masking initial keypoints on enhanced labels
    initial_label = F.conv2d(label_point_positions, kernel,
                             stride=1, padding=(kernel.shape[-1] - 1) // 2)
    initial_label[initial_label > 1] = 1

    if not PKE_learn:
        result = (loss_cal(detector_pred, initial_label.to(detector_pred)), 0, None, None, initial_label)
        return (*result, None) if return_stage_points else result

    nms_size = config['nms_size']
    nms_thresh = config['nms_thresh']
    scale = 8

    enhanced_label = None
    geometric_thresh = config['geometric_thresh']
    content_thresh = config['content_thresh']
    content_mode = config.get('pke_content_mode', 'one_way')
    weak_feedback = bool(config.get('pke_content_weak_feedback', False))
    strong_feedback_multiplier = int(config.get('pke_content_strong_feedback_multiplier', 1))
    weak_feedback_multiplier = int(config.get('pke_content_weak_feedback_multiplier', 1))
    with torch.no_grad():
        h, w = detector_pred.shape[2:]

        # number of learned points
        number_pts = 0
        points = nms(detector_pred, nms_thresh=nms_thresh, nms_size=nms_size,
                     detector_label=initial_label, mask=True)

        # geometric matching
        points, affine_points = mapping_points(grid_inverse, points, h, w)
        geo_points, affine_geo_points = geometric_filter(affine_detector_pred, points, affine_points,
                                                         geometric_thresh=geometric_thresh,
                                                         vessel_masks=vessel_masks,
                                                         relaxed_non_core_thresh=relaxed_non_core_thresh)


        # content matching
        content_points, affine_contend_points, content_feedback_weights = content_filter(
            descriptor_pred, affine_descriptor_pred, geo_points, affine_geo_points,
            content_thresh=content_thresh, scale=scale, mode=content_mode,
            weak_feedback=weak_feedback, strong_feedback_multiplier=strong_feedback_multiplier,
            weak_feedback_multiplier=weak_feedback_multiplier,
            return_feedback_weights=True,
        )
        enhanced_label_pts = []
        value_map_points = []
        for step in range(len(content_points)):
            # used to combine initial points and learned points
            positions = torch.where(label_point_positions[step, 0] == 1)
            if len(positions) == 2:
                positions = torch.cat((positions[1].unsqueeze(-1), positions[0].unsqueeze(-1)), -1)
            else:
                positions = positions[0]

            final_points = update_value_map(
                value_map[step], content_points[step], config,
                point_weights=content_feedback_weights[step],
            )
            value_map_points.append(final_points.detach().clone())

            # final_points = torch.cat((final_points, positions))

            temp_label = torch.zeros([h, w]).to(detector_pred.device)

            temp_label[final_points[:, 1], final_points[:, 0]] = 0.5
            temp_label[positions[:, 1], positions[:, 0]] = 1

            enhanced_kps = nms(temp_label.unsqueeze(0).unsqueeze(0), 0.1, 10)[0]
            if len(enhanced_kps) < len(positions):
                enhanced_kps = positions
            # print(len(final_points), len(positions), len(enhanced_kps))
            number_pts += (len(enhanced_kps) - len(positions))
            # number_pts += (len(enhanced_kps) - len(positions)) if (len(enhanced_kps) - len(positions)) > 0 else 0

            temp_label[:] = 0
            temp_label[enhanced_kps[:, 1], enhanced_kps[:, 0]] = 1

            enhanced_label_pts.append(temp_label.unsqueeze(0).unsqueeze(0))

            temp_label = F.conv2d(temp_label.unsqueeze(0).unsqueeze(0), kernel, stride=1,
                                  padding=(kernel.shape[-1] - 1) // 2)  # generating gaussian heatmaps
            temp_label[temp_label > 1] = 1

            if enhanced_label is None:
                enhanced_label = temp_label
            else:
                enhanced_label = torch.cat((enhanced_label, temp_label))

    enhanced_label_pts = torch.cat(enhanced_label_pts)
    affine_pred_inverse = F.grid_sample(affine_detector_pred, grid_inverse, align_corners=True)

    loss1 = loss_cal(detector_pred, enhanced_label)  # L_geo
    loss2 = loss_cal(detector_pred, affine_pred_inverse)  # L_clf
    # pred_mask = (enhanced_label > 0) & (affine_pred_inverse != 0)
    # loss2 = loss_cal(detector_pred[pred_mask], affine_pred_inverse[pred_mask])  # L_clf

    # mask_pred = grid_inverse
    # loss2 = loss_cal(detector_pred[mask_pred], affine_pred_inverse[mask_pred])  # L_clf

    loss = loss1+loss2

    result = (loss, number_pts, value_map, enhanced_label_pts, enhanced_label)
    if not return_stage_points:
        return result
    def copy_stage_points(point):
        # content_filter uses [] (rather than an empty Tensor) when a sample
        # has no valid content correspondence.  Diagnostics must preserve that
        # valid zero-count case without changing the PKE decision itself.
        if torch.is_tensor(point):
            return point.detach().clone()
        return torch.empty((0, 2), dtype=torch.long, device=detector_pred.device)

    stage_points = {
        'detector_candidates': [copy_stage_points(point) for point in points],
        'geometric_pass': [copy_stage_points(point) for point in geo_points],
        'content_pass': [copy_stage_points(point) for point in content_points],
        'value_map_points': [copy_stage_points(point) for point in value_map_points],
    }
    return (*result, stage_points)
