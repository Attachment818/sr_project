import argparse

import numpy as np
from tqdm import tqdm

from common.eval_util import compute_auc_from_errors, list_fimd_pairs, scale_reference_gt_to_query_space
from common.paper_viz import (
    save_match_image_v2,
    create_checkerboard_mosaic,
    draw_gt_on_checkerboard,
    save_overlay,
    save_dataset_success_curve,
)
from common.inference_diagnostics import write_jsonl
from common.spatial_geometry import estimate_homography_with_spatial_support
from predictor import Predictor
import os
import cv2
import yaml

# 支持命令行参数指定配置文件路径
parser = argparse.ArgumentParser(description='Test SuperRetina on FIMD dataset')
parser.add_argument('--config', type=str, default='./config/test_FIMD.yaml',
                    help='Path to config file (default: ./config/test_FIMD.yaml)')
parser.add_argument('--save_dir', type=str, default=None,
                    help='Directory to save results (default: sr/res/<checkpoint_stem>)')
args = parser.parse_args()

config_path = args.config
if os.path.exists(config_path):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError(f"Config File doesn't Exist: {config_path}")

Pred = Predictor(config)
Pred.set_eye_mask(None)

# 设置保存目录
if args.save_dir is not None:
    save_dir = args.save_dir
else:
    model_stem = os.path.splitext(os.path.basename(config['PREDICT']['model_save_path']))[0]
    save_dir = f'/home/data1/zhangjunhong/sr_project/sr/res/{model_stem}'
os.makedirs(save_dir, exist_ok=True)
save_inference_diagnostics = config['PREDICT'].get('save_inference_diagnostics', False)
diagnostics_path = os.path.join(save_dir, 'inference_diagnostics.jsonl')
if save_inference_diagnostics and os.path.exists(diagnostics_path):
    os.remove(diagnostics_path)

fimd_root = config.get('FIMD', {}).get(
    'data_root',
    os.path.join(os.path.dirname(__file__), 'data', 'FIMD'),
)
if not os.path.isabs(fimd_root):
    fimd_root = os.path.abspath(os.path.join(os.path.dirname(__file__), fimd_root))

use_matching_trick = config['PREDICT']['use_matching_trick']
displacement_threshold = config['PREDICT'].get('displacement_threshold', 300.0)
spatial_support_config = config['PREDICT'].get('spatial_support', {})
pair_items = list_fimd_pairs(fimd_root)
print(f"Loaded {len(pair_items)} FIMD pairs from: {fimd_root}")
big_num = 1e6
good_nums_rate = []
image_num = 0

failed = 0
inaccurate = 0
mae = 0
mee = 0

pair_avg_error_record = []

# 用于记录失败原因
failure_reasons = []

# 记录每对图像每对点的详细信息
pair_point_details = {}  # 存储每对图像的详细点信息，key为pair_name，value为点信息列表
# 记录每对图像每对点的误差
pair_errors = {}  # 存储每对图像的所有点误差，key为pair_name，value为误差列表
# 记录每对图像的平均误差
pair_avg_errors = {}  # 存储每对图像的平均误差，key为pair_name，value为平均误差

for pair_item in tqdm(pair_items):
    gt_file = pair_item['gt_file']
    file_name = pair_item['file_name']
    query_im_path = pair_item['query_im_path']
    refer_im_path = pair_item['refer_im_path']

    # 预先加载 GT 控制点（与 query/refer 同坐标系），供误差计算与论文可视化共用
    try:
        points_gd = np.loadtxt(gt_file)
        if points_gd.ndim == 1:
            points_gd = points_gd.reshape(1, -1)
        raw = np.zeros([len(points_gd), 2])
        dst = np.zeros([len(points_gd), 2])
        raw[:, 0] = points_gd[:, 2]
        raw[:, 1] = points_gd[:, 3]
        dst[:, 0] = points_gd[:, 0]
        dst[:, 1] = points_gd[:, 1]
        if config['PREDICT'].get('resize_refer_to_query', False):
            raw, dst = scale_reference_gt_to_query_space(
                raw, dst, query_im_path, refer_im_path
            )
    except Exception:
        raw = np.zeros((0, 2))
        dst = np.zeros((0, 2))
    
    # 从配置中读取逆一致性检查和异常值过滤参数
    use_inverse_consistency = config['PREDICT'].get('use_inverse_consistency', True)
    iccl = config['PREDICT'].get('iccl', 3.0)
    use_outlier_filter = config['PREDICT'].get('use_outlier_filter', True)
    outlier_criteria = config['PREDICT'].get('outlier_criteria', 'homography')
    outlier_threshold = config['PREDICT'].get('outlier_threshold', 20.0)
    
    # 调用带逆一致性检查和异常值过滤的匹配方法
    match_result = Pred.match_with_consistency_check(
        query_im_path, refer_im_path,
        use_inverse_consistency=use_inverse_consistency,
        iccl=iccl,
        use_outlier_filter=use_outlier_filter,
        outlier_criteria=outlier_criteria,
        outlier_threshold=outlier_threshold,
        return_diagnostics=save_inference_diagnostics,
    )
    if save_inference_diagnostics:
        goodMatch, cv_kpts_query, cv_kpts_refer, query_image, refer_image, match_diag = match_result
    else:
        goodMatch, cv_kpts_query, cv_kpts_refer, query_image, refer_image = match_result
        match_diag = None
    # 保存原始的goodMatch用于保存匹配图片
    goodMatch_for_save = goodMatch
    num_initial_matches = len(goodMatch) if goodMatch is not None else 0
    
    # 调试输出：显示关键点数量和匹配点数量
    num_query_keypoints = len(cv_kpts_query) if cv_kpts_query is not None else 0
    num_refer_keypoints = len(cv_kpts_refer) if cv_kpts_refer is not None else 0
    if num_initial_matches == 0:
        print(f"\n[调试] {file_name}: 检测到 {num_query_keypoints} 个query关键点, {num_refer_keypoints} 个refer关键点, 但匹配点数量为 0")
        print(f"      图片中显示的是所有检测到的关键点（蓝色=未匹配，红色+绿色连线=匹配点）")
    
    # 根据选择的配准方式计算变换矩阵
    H_m1 = None
    H_m2 = None
    quadratic_coeffs1 = None
    quadratic_coeffs2 = None
    mask = None
    stage1_inliers_rate = 0
    num_inliers = 0
    inliers_num_rate = 0
    num_inliers_stage2 = 0
    inliers_num_rate_stage2 = 0
    spatial_support_diag = None
    
    # 动态选择的配准方法（根据初始位移决定）
    selected_method = None
    
    # 先计算初始位移，根据位移阈值选择配准方法
    # 重要：src_pts 是 query 图像中的点，dst_pts 是 refer 图像中的点
    # 单应性矩阵 H 将 query 图像中的点变换到 refer 图像的坐标系
    if len(goodMatch) >= 4:  # 至少需要4个点来计算位移
        src_pts = [cv_kpts_query[m.queryIdx].pt for m in goodMatch]  # query 图像中的点
        dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in goodMatch]   # refer 图像中的点
        
        # 计算所有匹配点的位移
        displacements = []
        for src_pt, dst_pt in zip(src_pts, dst_pts):
            dx = dst_pt[0] - src_pt[0]
            dy = dst_pt[1] - src_pt[1]
            displacement = np.sqrt(dx**2 + dy**2)
            displacements.append(displacement)
        
        # 计算平均位移
        avg_displacement = np.mean(displacements) if displacements else 0.0
        
        # 根据位移阈值选择配准方法
        if avg_displacement > displacement_threshold:
            selected_method = 'quadratic'  # 位移大，使用二次多项式
        else:
            selected_method = 'homography'  # 位移小，使用单应性矩阵
        
        # 根据选择的配准方法计算变换矩阵
        min_points_required = 4 if selected_method == 'homography' else 6
        
        if len(goodMatch) >= min_points_required:
            if selected_method == 'homography':
                # 使用单应性变换
                # 注意：cv2.findHomography(src_pts, dst_pts) 计算从 src_pts 到 dst_pts 的变换矩阵
                # 即：H * src_pt = dst_pt（在齐次坐标系下）
                # 所以 H 可以将 query 图像中的点变换到 refer 图像的坐标系
                src_pts_array = np.float32(src_pts).reshape(-1, 1, 2)  # query 图像中的点
                dst_pts_array = np.float32(dst_pts).reshape(-1, 1, 2)  # refer 图像中的点
                H_m1, spatial_mask, spatial_support_diag = estimate_homography_with_spatial_support(
                    goodMatch, cv_kpts_query, cv_kpts_refer, query_image.shape,
                    enabled=spatial_support_config.get('enabled', False),
                    grid_size=int(spatial_support_config.get('grid_size', 4)),
                    max_per_cell=int(spatial_support_config.get('max_per_cell', 3)),
                    reprojection_threshold=float(spatial_support_config.get('reprojection_threshold', 20.0)),
                    min_inlier_retention=float(spatial_support_config.get('min_inlier_retention', 0.9)),
                    min_coverage_gain=int(spatial_support_config.get('min_coverage_gain', 1)),
                )
                mask = None if spatial_mask is None else spatial_mask.astype(np.uint8).reshape(-1, 1)
                if H_m1 is not None:
                    num_inliers = int(mask.sum())
                    inliers_num_rate = num_inliers / len(mask.ravel())
                    stage1_inliers_rate = inliers_num_rate
                    goodMatch = np.array(goodMatch)[mask.ravel() == 1]
                else:
                    inliers_num_rate = 0
                    num_inliers = 0
            else:
                # 使用二阶多项式变换
                try:
                    quadratic_coeffs1 = Pred.compute_quadratic_matrix(src_pts, dst_pts)
                    inliers_num_rate = 1.0  # 最小二乘法所有点都参与
                    num_inliers = len(goodMatch)
                    stage1_inliers_rate = inliers_num_rate
                except Exception as e:
                    print(f"Warning: Error computing quadratic matrix: {e}")
                    quadratic_coeffs1 = None
                    inliers_num_rate = 0
                    num_inliers = 0
        else:
            inliers_num_rate = 0
            num_inliers = 0
    
    # Matching trick 仅支持单应性变换
    if use_matching_trick and selected_method == 'homography':
        if H_m1 is not None:
            h, w = Pred.image_height, Pred.image_width
            query_align_first = cv2.warpPerspective(query_image, H_m1, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0))
            query_align_first = query_align_first.astype(float)
            query_align_first /= 255.
            H_m2, inliers_num_rate_stage2, _, _ = Pred.compute_homography(query_align_first, refer_im_path, query_is_image=True)
            if H_m2 is not None:
                inliers_num_rate = inliers_num_rate_stage2

    good_nums_rate.append(inliers_num_rate)
    image_num += 1

    # 保存匹配图（增强版）：兼容旧目录 + paper_figures
    if goodMatch_for_save is not None and len(goodMatch_for_save) > 0:
        match_img_dir = os.path.join(save_dir, "match_images")
        os.makedirs(match_img_dir, exist_ok=True)
        save_match_image_v2(
            goodMatch_for_save,
            cv_kpts_query,
            cv_kpts_refer,
            query_image,
            refer_image,
            os.path.join(match_img_dir, f"{file_name}_matches.png"),
        )
        paper_dir_m = os.path.join(save_dir, "paper_figures", file_name)
        os.makedirs(paper_dir_m, exist_ok=True)
        save_match_image_v2(
            goodMatch_for_save,
            cv_kpts_query,
            cv_kpts_refer,
            query_image,
            refer_image,
            os.path.join(paper_dir_m, "match.png"),
        )
    
    # 保存配准结果（仅在配准成功时保存）
    registration_success = False
    if selected_method == 'homography':
        registration_success = (inliers_num_rate >= 1e-6 and H_m1 is not None)
    elif selected_method == 'quadratic':
        registration_success = (inliers_num_rate >= 1e-6 and quadratic_coeffs1 is not None)
    else:
        registration_success = False
    
    if registration_success:
        # 获取图像尺寸
        h, w = refer_image.shape[:2]

        # 根据配准方式配准 query 到 refer 坐标系（可视化与算法均用 Predictor 灰度图）
        if selected_method == 'homography':
            # 确定最终的单应性矩阵
            H_final = H_m1
            if H_m2 is not None:
                # 如果使用matching trick，需要组合两个单应性矩阵
                H_final = np.dot(H_m2, H_m1)

            query_aligned = cv2.warpPerspective(
                query_image,
                H_final,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0),
            )
        else:
            # 前向系数为 query->refer；图像 warp 需 refer->query 的反向多项式 + inverse_map（与 backend 一致）
            quadratic_coeffs_inv = Pred.compute_quadratic_matrix(dst_pts, src_pts)
            query_aligned = Pred.warp_image_quadratic_inverse_map(
                query_image, quadratic_coeffs_inv, out_h=h, out_w=w
            )
            query_aligned = np.clip(query_aligned, 0, 255).astype(np.uint8)

        # 创建BGR合并图像（OpenCV使用BGR顺序：索引0=蓝色，索引1=绿色，索引2=红色）
        # query在蓝色通道（索引0），refer在绿色通道（索引1）
        merged = np.zeros((h, w, 3), dtype=np.uint8)
        if len(query_aligned.shape) == 3:
            merged[:, :, 0] = cv2.cvtColor(query_aligned, cv2.COLOR_BGR2GRAY)
        else:
            merged[:, :, 0] = query_aligned
        if len(refer_image.shape) == 3:
            merged[:, :, 1] = cv2.cvtColor(refer_image, cv2.COLOR_BGR2GRAY)
        else:
            merged[:, :, 1] = refer_image
        
        # 保存配准结果
        aligned_dir = os.path.join(save_dir, 'aligned_images')
        os.makedirs(aligned_dir, exist_ok=True)
        
        # 保存合并图像（用于可视化）
        merged_path = os.path.join(aligned_dir, f'{file_name}_aligned.png')
        cv2.imwrite(merged_path, merged)
        
        # 保存单独的配准后的query图像
        query_aligned_path = os.path.join(aligned_dir, f'{file_name}_query_aligned.png')
        cv2.imwrite(query_aligned_path, query_aligned)

        # 创建并保存棋盘格马赛克拼图
        # 从配置中读取棋盘格大小，默认32像素
        checkerboard_tile_size = config['PREDICT'].get('checkerboard_tile_size', 32)
        checkerboard = create_checkerboard_mosaic(
            query_aligned, refer_image, tile_size=checkerboard_tile_size
        )
        checkerboard_path = os.path.join(aligned_dir, f'{file_name}_checkerboard.png')
        cv2.imwrite(checkerboard_path, checkerboard)

        # 额外保存：用于对比验证
        checkerboard_original = create_checkerboard_mosaic(
            query_image, refer_image, tile_size=checkerboard_tile_size
        )
        checkerboard_original_path = os.path.join(aligned_dir, f'{file_name}_checkerboard_original.png')
        cv2.imwrite(checkerboard_original_path, checkerboard_original)

        if query_aligned.shape[:2] == query_image.shape[:2]:
            checkerboard_query_only = create_checkerboard_mosaic(
                query_image, query_aligned, tile_size=checkerboard_tile_size
            )
            checkerboard_query_only_path = os.path.join(
                aligned_dir, f'{file_name}_checkerboard_query_only.png'
            )
            cv2.imwrite(checkerboard_query_only_path, checkerboard_query_only)

        # ---------- 论文用：带 GT 的棋盘格、半透明叠加 ----------
        try:
            if len(raw) > 0:
                paper_dir = os.path.join(save_dir, "paper_figures", file_name)
                os.makedirs(paper_dir, exist_ok=True)
                if selected_method == "homography":
                    H_final = H_m1
                    if H_m2 is not None:
                        H_final = np.dot(H_m2, H_m1)
                    dst_pred_viz = cv2.perspectiveTransform(
                        raw.reshape(-1, 1, 2).astype(np.float32), H_final
                    ).squeeze()
                else:
                    raw_list = [(raw[i, 0], raw[i, 1]) for i in range(len(raw))]
                    dst_pred_viz = np.array(
                        Pred.transform_points_quadratic(raw_list, quadratic_coeffs1)
                    )
                cb = create_checkerboard_mosaic(
                    query_aligned, refer_image, tile_size=checkerboard_tile_size
                )
                cb_gt = draw_gt_on_checkerboard(cb, dst, dst_pred_viz)
                cv2.imwrite(os.path.join(paper_dir, "checkerboard_gt.png"), cb_gt)
                cv2.imwrite(os.path.join(paper_dir, "checkerboard.png"), cb)
                save_overlay(
                    query_aligned,
                    refer_image,
                    os.path.join(paper_dir, "overlay.png"),
                    alpha=0.5,
                    pts_ref=dst,
                    pts_pred=dst_pred_viz,
                    marker_radius=8,
                    marker_line_thickness=1,
                    marker_circle_outline=2,
                )
        except Exception as ex:
            print(f"[Warn] paper figure export failed for {file_name}: {ex}")
    
    failure_reason = None
    if not registration_success:
        failed += 1
        avg_dist = big_num
        # 记录失败情况和原因
        failure_reason = f"{file_name}: "
        if selected_method is None:
            min_points_needed = 4
            failure_reason += f"匹配点不足 (仅{num_initial_matches}个, 需要至少{min_points_needed}个用于计算初始位移)"
        elif num_initial_matches < (4 if selected_method == 'homography' else 6):
            min_points_required = 4 if selected_method == 'homography' else 6
            failure_reason += f"匹配点不足 (仅{num_initial_matches}个, 需要至少{min_points_required}个用于{selected_method})"
        elif selected_method == 'homography' and H_m1 is None:
            failure_reason += f"第一阶段单应性估计失败 (初始匹配点: {num_initial_matches}个)"
        elif selected_method == 'quadratic' and quadratic_coeffs1 is None:
            failure_reason += f"第一阶段二阶多项式估计失败 (初始匹配点: {num_initial_matches}个)"
        elif use_matching_trick and selected_method == 'homography' and H_m1 is not None and H_m2 is None:
            failure_reason += f"第二阶段单应性估计失败 (第一阶段内点比例: {stage1_inliers_rate*100:.2f}%, 初始匹配点: {num_initial_matches}个)"
        else:
            failure_reason += f"内点比例过低 ({inliers_num_rate*100:.4f}%, 阈值: 0.0001%, 初始匹配点: {num_initial_matches}个, 内点数: {num_inliers}个)"
        
        failure_reasons.append(failure_reason)
        if len(failure_reasons) <= 10 or failed % 10 == 0:  # 只打印前10个和每10个失败案例
            print(f"  ✗ {failure_reason}")
        
        pair_avg_errors[file_name] = avg_dist
        pair_errors[file_name] = []
        pair_point_details[file_name] = []
    else:
        # raw / dst 已在循环开头由 GT 文件加载
        # 根据配准方式计算预测点
        if selected_method == 'homography':
            # 使用单应性变换
            raw_points_list = [(raw[i, 0], raw[i, 1]) for i in range(len(raw))]
            dst_pred = cv2.perspectiveTransform(raw.reshape(-1, 1, 2), H_m1)
            if H_m2 is not None:
                dst_pred = cv2.perspectiveTransform(dst_pred.reshape(-1, 1, 2), H_m2)
            dst_pred = dst_pred.squeeze()
        else:
            # 使用二阶多项式变换
            raw_points_list = [(raw[i, 0], raw[i, 1]) for i in range(len(raw))]
            transformed_points = Pred.transform_points_quadratic(raw_points_list, quadratic_coeffs1)
            dst_pred = np.array(transformed_points)

        dis = (dst - dst_pred) ** 2
        dis = np.sqrt(dis[:, 0] + dis[:, 1])
        avg_dist = dis.mean()
        
        # 记录每对点的详细信息
        point_details = []
        for i in range(len(raw)):
            point_info = {
                'raw_point': [float(raw[i, 0]), float(raw[i, 1])],
                'dst_point': [float(dst[i, 0]), float(dst[i, 1])],
                'pred_point': [float(dst_pred[i, 0]), float(dst_pred[i, 1])],
                'error': float(dis[i])
            }
            point_details.append(point_info)
        pair_point_details[file_name] = point_details
        
        # 记录该对图像所有点的误差
        pair_errors[file_name] = dis.tolist()
        # 记录平均误差
        pair_avg_errors[file_name] = avg_dist
        
        mae = dis.max()
        mee = np.median(dis)
        if mae > 50 or mee > 20:
            inaccurate += 1

    if match_diag is not None:
        match_diag.update({
            'pair_id': file_name,
            'dataset': 'FIMD',
            'selected_method': selected_method,
            'matches_after_consistency': int(num_initial_matches),
            'geometry_inliers': int(num_inliers),
            'geometry_inlier_rate': float(inliers_num_rate),
            'stage1_inlier_rate': float(stage1_inliers_rate),
            'matching_trick_used': bool(H_m2 is not None),
            'registration_success': bool(registration_success),
            'failure_reason': failure_reason,
            'average_control_point_error': float(avg_dist),
        })
        if spatial_support_diag is not None:
            match_diag.update(spatial_support_diag)
        write_jsonl(diagnostics_path, match_diag)
    
    pair_avg_error_record.append(avg_dist)

print('-'*40)
print(f"Failed:{'%.2f' % (100*failed/image_num)}%, Inaccurate:{'%.2f' % (100*inaccurate/image_num)}%, "
      f"Acceptable:{'%.2f' % (100*(image_num-inaccurate-failed)/image_num)}%")

print('-'*40)

# 输出失败原因统计
if len(failure_reasons) > 0:
    print(f"\n失败案例统计 (共{len(failure_reasons)}个失败案例):")
    # 统计失败原因类型
    reason_counts = {}
    for reason in failure_reasons:
        # 提取失败原因类型（冒号后的关键词）
        reason_text = reason.split(': ', 1)[1] if ': ' in reason else reason
        reason_type = reason_text.split('(')[0].strip()
        reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
    
    print("失败原因分布:")
    for reason_type, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason_type}: {count} 个 ({count/len(failure_reasons)*100:.1f}%)")
    
    if len(failure_reasons) <= 20:
        print(f"\n所有失败案例详情:")
        for reason in failure_reasons:
            print(f"  {reason}")
    else:
        print(f"\n前10个失败案例:")
        for reason in failure_reasons[:10]:
            print(f"  {reason}")
        print(f"  ... 还有 {len(failure_reasons)-10} 个失败案例")
    print('-'*40)

auc = compute_auc_from_errors(pair_avg_error_record)
print('Pairs: %d, AUC: %.3f' % (len(pair_avg_error_record), auc))

curve_path = os.path.join(save_dir, "paper_figures", "FIMD_success_rate_vs_threshold.png")
save_dataset_success_curve(pair_avg_error_record, curve_path, dataset_name="FIMD", threshold_max=25)
print(f"Success-rate curve saved to: {curve_path}")

# 保存每对点的详细信息到文本文件
if len(pair_avg_errors) > 0:
    errors_output_path = os.path.join(save_dir, 'FIMD_point_errors.txt')
    with open(errors_output_path, 'w', encoding='utf-8') as f:
        for pair_name in sorted(pair_avg_errors.keys()):
            # 写入图像对名称
            f.write(f"{pair_name}\n")
            # 写入该对图像所有点的误差（逗号分隔，如果存在）
            if pair_name in pair_errors and len(pair_errors[pair_name]) > 0:
                errors_str = ','.join([f"{err:.4f}" for err in pair_errors[pair_name]])
                f.write(f"errors:{errors_str}\n")
            # 写入该对图像的平均误差
            avg_error = pair_avg_errors[pair_name]
            f.write(f"avg_error:{avg_error:.4f}\n")
            # 写入每对点的详细信息
            if pair_name in pair_point_details and len(pair_point_details[pair_name]) > 0:
                f.write("point_details:\n")
                for idx, point_info in enumerate(pair_point_details[pair_name]):
                    f.write(f"  point_{idx}: raw=({point_info['raw_point'][0]:.2f},{point_info['raw_point'][1]:.2f}) "
                           f"dst=({point_info['dst_point'][0]:.2f},{point_info['dst_point'][1]:.2f}) "
                           f"pred=({point_info['pred_point'][0]:.2f},{point_info['pred_point'][1]:.2f}) "
                           f"error={point_info['error']:.4f}\n")
            f.write("\n")
    
    total_points = sum(len(errors) for errors in pair_errors.values())
    print(f'\n每对点的误差已保存到: {errors_output_path}')
    print(f'总共记录了 {len(pair_errors)} 对图像，{total_points} 个点的误差信息')
    print(f'匹配图片已保存到: {os.path.join(save_dir, "match_images")} 与 {os.path.join(save_dir, "paper_figures")}')
    print(f'配准结果已保存到: {os.path.join(save_dir, "aligned_images")}')
