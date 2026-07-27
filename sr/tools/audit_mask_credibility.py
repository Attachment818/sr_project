"""Audit pseudo-vessel mask stability without claiming segmentation accuracy."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.common_util import pre_processing


OUTPUT_NAMES = (
    'mask_stability_per_image.csv',
    'mask_variant_sensitivity.csv',
    'mask_credibility_summary.json',
    'mask_credibility_report.md',
)
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    return parser.parse_args()


def morph_mask(image, threshold=0.25, blackhat_kernel=15, dilate_kernel=3):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blackhat_kernel, blackhat_kernel))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)
    maximum = float(blackhat.max())
    normalized = blackhat.astype(np.float32) / maximum if maximum > 0 else blackhat.astype(np.float32)
    mask = (normalized >= threshold).astype(np.uint8)
    if dilate_kernel > 1:
        dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, dilate, iterations=1)
    return mask


def skeletonize(mask):
    current = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(current)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(current):
        opened = cv2.morphologyEx(current, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = cv2.erode(current, element)
    return (skeleton > 0).astype(np.uint8)


def junction_centres(skeleton):
    neighbors = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16)) - skeleton
    candidates = ((skeleton > 0) & (neighbors >= 3)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    return [tuple(map(float, centroids[index])) for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] > 0]


def binary_iou(first, second, valid=None):
    valid = np.ones(first.shape, bool) if valid is None else valid.astype(bool)
    first = first.astype(bool) & valid
    second = second.astype(bool) & valid
    union = np.count_nonzero(first | second)
    return 1.0 if union == 0 else float(np.count_nonzero(first & second) / union)


def skeleton_f1(first, second, tolerance=2, valid=None):
    valid = np.ones(first.shape, bool) if valid is None else valid.astype(bool)
    first = first.astype(bool) & valid
    second = second.astype(bool) & valid
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    first_dilated = cv2.dilate(first.astype(np.uint8), kernel) > 0
    second_dilated = cv2.dilate(second.astype(np.uint8), kernel) > 0
    precision = np.count_nonzero(first & second_dilated) / max(1, np.count_nonzero(first))
    recall = np.count_nonzero(second & first_dilated) / max(1, np.count_nonzero(second))
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def junction_repeatability(first, second, radius=5.0):
    if not first or not second:
        return 1.0 if not first and not second else 0.0
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    distance = np.linalg.norm(a[:, None] - b[None], axis=2)
    forward = float((distance.min(axis=1) <= radius).mean())
    backward = float((distance.min(axis=0) <= radius).mean())
    return 0.5 * (forward + backward)


def read_green_preprocessed(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unable to read image: {path}')
    green = image[:, :, 1]
    green = pre_processing(green)
    green = np.clip(green * 255.0 if green.max() <= 1.0 else green, 0, 255).astype(np.uint8)
    return cv2.resize(green, tuple(size), interpolation=cv2.INTER_AREA)


def affine_matrix(width, height, angle, scale, dx, dy):
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += (dx, dy)
    return matrix


def restore_affine_mask(mask, matrix, width, height):
    inverse = cv2.invertAffineTransform(matrix)
    restored = cv2.warpAffine(mask, inverse, (width, height), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    transformed_valid = cv2.warpAffine(np.ones((height, width), np.uint8), matrix, (width, height),
                                       flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
    valid = cv2.warpAffine(transformed_valid, inverse, (width, height), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return restored, valid > 0


def photometric_variants(image):
    return {
        'brightness_minus30': np.clip(image.astype(np.int16) - 30, 0, 255).astype(np.uint8),
        'brightness_plus30': np.clip(image.astype(np.int16) + 30, 0, 255).astype(np.uint8),
        'contrast_070': np.clip((image.astype(np.float32) - 127.5) * 0.70 + 127.5, 0, 255).astype(np.uint8),
        'contrast_130': np.clip((image.astype(np.float32) - 127.5) * 1.30 + 127.5, 0, 255).astype(np.uint8),
        'gaussian_blur': cv2.GaussianBlur(image, (7, 7), 1.5),
    }


def list_images(root, maximum, seed):
    candidates = sorted(path for path in root.rglob('*') if path.suffix.lower() in IMAGE_SUFFIXES)
    if not candidates:
        raise FileNotFoundError(f'No images found under: {root}')
    if maximum > 0 and len(candidates) > maximum:
        rng = random.Random(seed)
        candidates = sorted(rng.sample(candidates, maximum))
    return candidates


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def self_test():
    image = np.full((128, 128), 180, np.uint8)
    cv2.line(image, (16, 64), (112, 64), 20, 3)
    cv2.line(image, (64, 64), (64, 112), 20, 3)
    mask = morph_mask(image)
    skeleton = skeletonize(mask)
    assert mask.shape == image.shape and mask.dtype == np.uint8
    assert 0.99 <= binary_iou(mask, mask) <= 1.0
    assert skeleton_f1(skeleton, skeleton) > 0.99
    assert junction_repeatability([(10.0, 10.0)], [(12.0, 10.0)], radius=3.0) == 1.0
    print('mask credibility self-test passed')


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_config is None:
        raise ValueError('--audit-config is required unless --self-test is used')
    audit = yaml.safe_load(args.audit_config.read_text(encoding='utf-8'))['AUDIT']
    output_dir = Path(audit['output_dir'])
    generated = [output_dir / name for name in OUTPUT_NAMES]
    occupied = [path for path in generated if path.exists()]
    if occupied:
        raise FileExistsError('Refusing to overwrite mask audit result(s): ' + ', '.join(map(str, occupied)))
    output_dir.mkdir(parents=True, exist_ok=True)
    size = [int(value) for value in audit.get('resize', [768, 768])]
    if len(size) != 2 or min(size) <= 0:
        raise ValueError('resize must contain two positive integers: [width, height]')
    baseline = audit.get('baseline_mask', {})
    base_threshold = float(baseline.get('threshold', 0.25))
    base_kernel = int(baseline.get('blackhat_kernel', 15))
    base_dilate = int(baseline.get('dilate_kernel', 3))
    variant_specs = audit.get('mask_variants', [])
    transforms = audit.get('affine_transforms', [])
    if not variant_specs or not transforms:
        raise ValueError('mask_variants and affine_transforms must be non-empty')

    stability_rows, variant_rows = [], []
    for dataset_index, dataset in enumerate(audit.get('datasets', [])):
        root = Path(dataset['root'])
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found for {dataset['name']}: {root}")
        images = list_images(root, int(dataset.get('max_images', 0)),
                             int(audit.get('seed', 3407)) + dataset_index)
        for image_path in tqdm(images, desc=f"mask audit {dataset['name']}", unit='image'):
            image = read_green_preprocessed(image_path, size)
            height, width = image.shape
            reference = morph_mask(image, base_threshold, base_kernel, base_dilate)
            reference_skeleton = skeletonize(reference)
            reference_junctions = junction_centres(reference_skeleton)

            for spec in variant_specs:
                candidate = morph_mask(image, float(spec['threshold']),
                                       int(spec['blackhat_kernel']), int(spec['dilate_kernel']))
                candidate_skeleton = skeletonize(candidate)
                variant_rows.append({
                    'dataset': dataset['name'], 'image': image_path.name, 'variant': spec['name'],
                    'mask_iou_to_baseline': binary_iou(reference, candidate),
                    'skeleton_f1_to_baseline': skeleton_f1(reference_skeleton, candidate_skeleton),
                    'junction_repeatability_to_baseline': junction_repeatability(
                        reference_junctions, junction_centres(candidate_skeleton),
                        radius=float(audit.get('junction_radius', 5.0))),
                    'baseline_area_fraction': float(reference.mean()),
                    'variant_area_fraction': float(candidate.mean()),
                    'baseline_junction_count': len(reference_junctions),
                    'variant_junction_count': len(junction_centres(candidate_skeleton)),
                })

            for name, changed in photometric_variants(image).items():
                candidate = morph_mask(changed, base_threshold, base_kernel, base_dilate)
                candidate_skeleton = skeletonize(candidate)
                stability_rows.append({
                    'dataset': dataset['name'], 'image': image_path.name,
                    'perturbation_type': 'photometric', 'perturbation': name,
                    'mask_iou': binary_iou(reference, candidate),
                    'skeleton_f1': skeleton_f1(reference_skeleton, candidate_skeleton),
                    'junction_repeatability': junction_repeatability(
                        reference_junctions, junction_centres(candidate_skeleton),
                        radius=float(audit.get('junction_radius', 5.0))),
                    'valid_fraction': 1.0,
                })

            for transform in transforms:
                matrix = affine_matrix(width, height, float(transform['angle']),
                                       float(transform['scale']), float(transform['dx']),
                                       float(transform['dy']))
                warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                warped_mask = morph_mask(warped, base_threshold, base_kernel, base_dilate)
                restored, valid = restore_affine_mask(warped_mask, matrix, width, height)
                restored_skeleton = skeletonize(restored & valid.astype(np.uint8))
                stability_rows.append({
                    'dataset': dataset['name'], 'image': image_path.name,
                    'perturbation_type': 'affine', 'perturbation': transform['name'],
                    'mask_iou': binary_iou(reference, restored, valid),
                    'skeleton_f1': skeleton_f1(reference_skeleton, restored_skeleton, valid=valid),
                    'junction_repeatability': junction_repeatability(
                        junction_centres(reference_skeleton & valid.astype(np.uint8)),
                        junction_centres(restored_skeleton),
                        radius=float(audit.get('junction_radius', 5.0))),
                    'valid_fraction': float(valid.mean()),
                })

    if not stability_rows or not variant_rows:
        raise RuntimeError('Mask audit produced no rows')
    with generated[0].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=stability_rows[0].keys())
        writer.writeheader(); writer.writerows(stability_rows)
    with generated[1].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=variant_rows[0].keys())
        writer.writeheader(); writer.writerows(variant_rows)

    summary = {'scope': 'stability_only_no_vessel_ground_truth', 'datasets': {}}
    for dataset in sorted({row['dataset'] for row in stability_rows}):
        summary['datasets'][dataset] = {}
        for kind in ('photometric', 'affine'):
            selected = [row for row in stability_rows
                        if row['dataset'] == dataset and row['perturbation_type'] == kind]
            summary['datasets'][dataset][kind] = {
                'rows': len(selected),
                'mask_iou_mean': mean_or_none([row['mask_iou'] for row in selected]),
                'skeleton_f1_mean': mean_or_none([row['skeleton_f1'] for row in selected]),
                'junction_repeatability_mean': mean_or_none(
                    [row['junction_repeatability'] for row in selected]),
            }
    summary['limitations'] = [
        'No pixel-level vessel ground truth is available in the configured datasets.',
        'The audit measures parameter and perturbation stability, not medical segmentation accuracy.',
        'Stable pseudo-junctions may still be anatomically false and can only justify soft supervision.',
    ]
    generated[2].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report = ['# Mask credibility audit', '',
              'Scope: stability and sensitivity only; no vessel ground truth was used.', '',
              '| Dataset | Perturbation | Mask IoU | Skeleton F1 | Junction repeatability |',
              '|---|---|---:|---:|---:|']
    for dataset, values in summary['datasets'].items():
        for kind, metrics in values.items():
            report.append('| {} | {} | {:.4f} | {:.4f} | {:.4f} |'.format(
                dataset, kind, metrics['mask_iou_mean'], metrics['skeleton_f1_mean'],
                metrics['junction_repeatability_mean']))
    report.extend(['', 'These values cannot be interpreted as segmentation Dice or anatomical accuracy.', ''])
    generated[3].write_text('\n'.join(report), encoding='utf-8')
    print(f'Wrote mask credibility audit: {output_dir}')


if __name__ == '__main__':
    main()
