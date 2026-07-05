import os

import cv2
import numpy as np


# Compute AUC scores for image registration on the FIRE dataset
def compute_auc(s_error, p_error, a_error):
    assert (len(s_error) == 71)  # Easy pairs
    assert (len(p_error) == 48)  # Hard pairs. Note file control_points_P37_1_2.txt is ignored
    assert (len(a_error) == 14)  # Moderate pairs

    s_error = np.array(s_error)
    p_error = np.array(p_error)
    a_error = np.array(a_error)

    limit = 25
    gs_error = np.zeros(limit + 1)
    gp_error = np.zeros(limit + 1)
    ga_error = np.zeros(limit + 1)

    accum_s = 0
    accum_p = 0
    accum_a = 0

    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
        gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
        ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

        accum_s = accum_s + gs_error[i]
        accum_p = accum_p + gp_error[i]
        accum_a = accum_a + ga_error[i]

    auc_s = accum_s / (limit * 100)
    auc_p = accum_p / (limit * 100)
    auc_a = accum_a / (limit * 100)
    mAUC = (auc_s + auc_p + auc_a) / 3.0
    return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mAUC}


def compute_auc_from_errors(errors, limit=25):
    """FIRE/FIMD protocol: AUC of pair success rate vs error threshold (1..limit px)."""
    errors = np.array(errors, dtype=np.float64)
    if errors.size == 0:
        return 0.0
    accum = 0.0
    for i in range(1, limit + 1):
        accum += np.sum(errors < i) * 100.0 / len(errors)
    return accum / (limit * 100)


def list_fimd_pairs(data_root):
    """Scan FIMD root and return sorted pair metadata."""
    pairs = []
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"FIMD data root not found: {data_root}")

    for folder_name in sorted(os.listdir(data_root)):
        pair_dir = os.path.join(data_root, folder_name)
        if not os.path.isdir(pair_dir):
            continue

        gt_file = os.path.join(pair_dir, f"control_points_{folder_name}.txt")
        if not os.path.isfile(gt_file):
            continue

        pair_id = folder_name.split("_")[0]
        refer_im_path = os.path.join(pair_dir, f"{pair_id}_r.jpg")
        query_im_path = os.path.join(pair_dir, f"{pair_id}_t.jpg")
        if not (os.path.isfile(refer_im_path) and os.path.isfile(query_im_path)):
            raise FileNotFoundError(
                f"Missing images for {folder_name}: {refer_im_path} / {query_im_path}"
            )

        pairs.append({
            "pair_name": folder_name,
            "file_name": f"control_points_{folder_name}",
            "gt_file": gt_file,
            "query_im_path": query_im_path,
            "refer_im_path": refer_im_path,
        })

    if not pairs:
        raise FileNotFoundError(f"No valid FIMD pairs found under: {data_root}")
    return pairs


def scale_reference_gt_to_query_space(raw, dst, query_im_path, refer_im_path):
    """FIMD: GT 中 reference 点坐标缩放到 query(t) 图像尺寸（与 resize refer 一致）。"""
    refer_img = cv2.imread(refer_im_path, cv2.IMREAD_COLOR)
    query_img = cv2.imread(query_im_path, cv2.IMREAD_COLOR)
    if refer_img is None or query_img is None:
        return raw, dst

    h_r, w_r = refer_img.shape[:2]
    h_t, w_t = query_img.shape[:2]
    if (h_r, w_r) == (h_t, w_t):
        return raw, dst

    dst_scaled = dst.copy()
    dst_scaled[:, 0] *= w_t / w_r
    dst_scaled[:, 1] *= h_t / h_r
    return raw, dst_scaled
