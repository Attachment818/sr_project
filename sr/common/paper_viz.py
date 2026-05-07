"""
论文级可视化：匹配图、棋盘格（含 GT 点）、叠加图、阈值–成功率曲线（配准中常称 accuracy / success-rate curve，非分类 ROC）。
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def save_match_image_v2(
    goodMatch,
    cv_kpts_query,
    cv_kpts_refer,
    query_image,
    refer_image,
    save_path: str,
    unmatched_radius: int = 3,
    matched_radius: int = 8,
    line_thickness: int = 2,
    circle_thickness: int = 2,
    match_line_double_layer: bool = True,
) -> None:
    """并排匹配图：未匹配点较小，匹配点更大更醒目。
    match_line_double_layer=True：黄底+橙色双线（更醒目）；False：仅一条细线（line_thickness，LINE_AA）。"""
    _d = os.path.dirname(save_path)
    if _d:
        os.makedirs(_d, exist_ok=True)

    if len(query_image.shape) == 2:
        query_image_color = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    else:
        query_image_color = query_image.copy()

    if len(refer_image.shape) == 2:
        refer_image_color = cv2.cvtColor(refer_image, cv2.COLOR_GRAY2BGR)
    else:
        refer_image_color = refer_image.copy()

    h1, w1 = query_image_color.shape[:2]
    h2, w2 = refer_image_color.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[0:h1, 0:w1] = query_image_color
    vis[0:h2, w1:] = refer_image_color

    if goodMatch is None or len(goodMatch) == 0:
        cv2.imwrite(save_path, vis)
        return

    good_list = goodMatch.tolist() if isinstance(goodMatch, np.ndarray) else list(goodMatch)

    matched_q = set(m.queryIdx for m in good_list)
    matched_r = set(m.trainIdx for m in good_list)

    # 未匹配：灰色小点
    for i, kp in enumerate(cv_kpts_query):
        if i not in matched_q:
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(vis, pt, unmatched_radius, (140, 140, 140), -1)

    for i, kp in enumerate(cv_kpts_refer):
        if i not in matched_r:
            pt = (int(round(kp.pt[0] + w1)), int(round(kp.pt[1])))
            cv2.circle(vis, pt, unmatched_radius, (140, 140, 140), -1)

    # 匹配：粗线 + 白边彩色圆（更醒目）
    for m in good_list:
        pt1 = (
            int(round(cv_kpts_query[m.queryIdx].pt[0])),
            int(round(cv_kpts_query[m.queryIdx].pt[1])),
        )
        pt2 = (
            int(round(cv_kpts_refer[m.trainIdx].pt[0] + w1)),
            int(round(cv_kpts_refer[m.trainIdx].pt[1])),
        )
        if match_line_double_layer:
            cv2.line(vis, pt1, pt2, (0, 255, 255), line_thickness + 2)  # 亮黄底
            cv2.line(vis, pt1, pt2, (0, 165, 255), line_thickness)  # 橙线
        else:
            cv2.line(
                vis,
                pt1,
                pt2,
                (0, 165, 255),
                max(1, line_thickness),
                lineType=cv2.LINE_AA,
            )
        for pt, col in ((pt1, (0, 255, 0)), (pt2, (255, 0, 255))):
            cv2.circle(vis, pt, matched_radius + 2, (255, 255, 255), circle_thickness)
            cv2.circle(vis, pt, matched_radius, col, -1)

    cv2.imwrite(save_path, vis)


def _resize_second_to_first(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """将 img2 缩放到与 img1 相同 HxW，保持像素一一对应（勿用裁剪，否则会错位）。"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) == (h2, w2):
        return img1, img2
    img2r = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LINEAR)
    return img1, img2r


def create_checkerboard_mosaic(
    img1: np.ndarray,
    img2: np.ndarray,
    tile_size: int = 32,
) -> np.ndarray:
    """灰度棋盘格；img1/img2 可为灰度或 BGR，内部转灰度。尺寸不一致时缩放对齐（与旧版裁剪不同）。"""
    img1, img2 = _resize_second_to_first(img1, img2)
    h, w = img1.shape[:2]

    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1.copy()
    if len(img2.shape) == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g2 = img2.copy()

    out = np.zeros((h, w), dtype=np.uint8)
    nh = (h + tile_size - 1) // tile_size
    nw = (w + tile_size - 1) // tile_size
    for i in range(nh):
        for j in range(nw):
            y0, y1 = i * tile_size, min((i + 1) * tile_size, h)
            x0, x1 = j * tile_size, min((j + 1) * tile_size, w)
            src = g1 if (i + j) % 2 == 0 else g2
            out[y0:y1, x0:x1] = src[y0:y1, x0:x1]
    return out


def create_checkerboard_mosaic_bgr(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    tile_size: int = 32,
) -> np.ndarray:
    """彩色棋盘格：两幅图须在参考系对齐；尺寸不一致时缩放 img2 到 img1。"""
    img1_bgr, img2_bgr = _resize_second_to_first(img1_bgr, img2_bgr)
    if len(img1_bgr.shape) == 2:
        img1_bgr = cv2.cvtColor(img1_bgr, cv2.COLOR_GRAY2BGR)
    if len(img2_bgr.shape) == 2:
        img2_bgr = cv2.cvtColor(img2_bgr, cv2.COLOR_GRAY2BGR)
    h, w = img1_bgr.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    nh = (h + tile_size - 1) // tile_size
    nw = (w + tile_size - 1) // tile_size
    for i in range(nh):
        for j in range(nw):
            y0, y1 = i * tile_size, min((i + 1) * tile_size, h)
            x0, x1 = j * tile_size, min((j + 1) * tile_size, w)
            src = img1_bgr if (i + j) % 2 == 0 else img2_bgr
            out[y0:y1, x0:x1] = src[y0:y1, x0:x1]
    return out


def draw_control_point_pairs_on_bgr(
    vis_bgr: np.ndarray,
    pts_ref: np.ndarray,
    pts_pred: np.ndarray,
    radius: int = 10,
    line_between: bool = True,
    line_thickness_between: int = 1,
    circle_outline_thickness: int = 2,
    draw_legend: bool = False,
) -> np.ndarray:
    """在 BGR 图上绘制参考 GT（绿）、配准预测（品红）及连线。pts_* 为 (N,2)。"""
    vis = vis_bgr.copy()
    n = min(len(pts_ref), len(pts_pred))
    for i in range(n):
        pr = (int(round(pts_ref[i, 0])), int(round(pts_ref[i, 1])))
        pf = (int(round(pts_pred[i, 0])), int(round(pts_pred[i, 1])))
        if line_between:
            cv2.line(
                vis,
                pr,
                pf,
                (0, 255, 255),
                line_thickness_between,
                lineType=cv2.LINE_AA,
            )
        ir = max(1, radius - 4)
        cv2.circle(vis, pr, radius, (0, 255, 0), circle_outline_thickness, lineType=cv2.LINE_AA)
        cv2.circle(vis, pr, ir, (0, 200, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, pf, radius, (255, 0, 255), circle_outline_thickness, lineType=cv2.LINE_AA)
        cv2.circle(vis, pf, ir, (200, 0, 200), -1, lineType=cv2.LINE_AA)
    if draw_legend:
        cv2.putText(
            vis,
            "Green: REF GT   Magenta: FLOAT (warped)",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "Green: REF GT   Magenta: FLOAT (warped)",
            (7, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    return vis


def draw_gt_on_checkerboard(
    checkerboard_gray: np.ndarray,
    pts_ref: np.ndarray,
    pts_float_warped_to_ref: np.ndarray,
    radius: int = 10,
    line_between: bool = True,
    line_thickness_between: int = 1,
    circle_outline_thickness: int = 2,
) -> np.ndarray:
    """
    在棋盘格（与参考图同尺寸、对齐后的坐标系）上绘制：
      - 参考图 GT 点（绿色）
      - 浮动图控制点经变换到参考系后的位置（品红）
      - 可选：对应点连线（黄色，便于看误差）
    pts_* 形状 (N,2)，与 checkerboard 同一像素坐标系。
    支持输入灰度或 BGR。
    """
    if len(checkerboard_gray.shape) == 3:
        vis = checkerboard_gray.copy()
    else:
        vis = cv2.cvtColor(checkerboard_gray, cv2.COLOR_GRAY2BGR)
    return draw_control_point_pairs_on_bgr(
        vis,
        pts_ref,
        pts_float_warped_to_ref,
        radius=radius,
        line_between=line_between,
        line_thickness_between=line_thickness_between,
        circle_outline_thickness=circle_outline_thickness,
        draw_legend=True,
    )


def save_overlay(
    img_a: np.ndarray,
    img_b: np.ndarray,
    save_path: str,
    alpha: float = 0.5,
    pts_ref: Optional[np.ndarray] = None,
    pts_pred: Optional[np.ndarray] = None,
    marker_radius: int = 8,
    marker_line_between: bool = True,
    marker_line_thickness: int = 1,
    marker_circle_outline: int = 2,
) -> None:
    """半透明叠加；可选在混合后绘制控制点（绿/品红/黄线，与棋盘格一致，无图例）。"""
    _d = os.path.dirname(save_path)
    if _d:
        os.makedirs(_d, exist_ok=True)
    if img_a.shape[:2] != img_b.shape[:2]:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(img_a.shape) == 2:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
    if len(img_b.shape) == 2:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
    blend = cv2.addWeighted(img_a, alpha, img_b, 1.0 - alpha, 0)
    if (
        pts_ref is not None
        and pts_pred is not None
        and len(pts_ref) > 0
        and len(pts_pred) > 0
    ):
        blend = draw_control_point_pairs_on_bgr(
            blend,
            pts_ref,
            pts_pred,
            radius=marker_radius,
            line_between=marker_line_between,
            line_thickness_between=marker_line_thickness,
            circle_outline_thickness=marker_circle_outline,
            draw_legend=False,
        )
    cv2.imwrite(save_path, blend)


def save_success_rate_curve(
    pair_errors: Sequence[float],
    save_path: str,
    title: str = "Registration",
    threshold_max: int = 25,
    ylabel: str = "Pair success rate (%)",
) -> None:
    """
    阈值–成功率曲线：对每个阈值 t，统计 E < t 的图像对比例（含失败对为 inf）。
    这是配准论文常用展示，不是二分类 ROC。
    """
    if plt is None:
        return
    _d = os.path.dirname(save_path)
    if _d:
        os.makedirs(_d, exist_ok=True)
    arr = np.array(pair_errors, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return
    xs = list(range(1, threshold_max + 1))
    ys = []
    for t in xs:
        ys.append(100.0 * np.sum(arr < t) / n)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(xs, ys, "b-", linewidth=2, label="Success rate")
    ax.set_xlabel("Error threshold (pixels)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: success rate vs threshold (N={n} pairs)")
    ax.set_xlim(1, threshold_max)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_fire_success_curves(
    auc_record: dict,
    save_path: str,
    threshold_max: int = 25,
) -> None:
    """FIRE：S/P/A 三条成功率曲线同图。"""
    if plt is None:
        return
    _d = os.path.dirname(save_path)
    if _d:
        os.makedirs(_d, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    colors = {"S": "green", "P": "orange", "A": "red"}
    for cat in ("S", "P", "A"):
        errs = np.array(auc_record.get(cat, []), dtype=np.float64)
        if errs.size == 0:
            continue
        n = len(errs)
        xs = list(range(1, threshold_max + 1))
        ys = [100.0 * np.sum(errs < t) / n for t in xs]
        ax.plot(xs, ys, color=colors.get(cat, "blue"), linewidth=2, label=f"Category {cat} (N={n})")
    ax.set_xlabel("Error threshold (pixels)")
    ax.set_ylabel("Pair success rate (%)")
    ax.set_title("FIRE: success rate vs threshold by category")
    ax.set_xlim(1, threshold_max)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
