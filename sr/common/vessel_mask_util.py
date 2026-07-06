"""Online vessel mask generation for Phase 1A region-selective vessel loss."""

import cv2
import numpy as np
import torch


def _to_gray_uint8(image_np: np.ndarray) -> np.ndarray:
    """(H, W) or (C, H, W) float/uint8 -> (H, W) uint8."""
    if image_np.ndim == 3:
        if image_np.shape[0] in (1, 3):
            if image_np.shape[0] == 3:
                image_np = image_np[1]  # green channel, consistent with Lab4 pipeline
            else:
                image_np = image_np[0]
        else:
            image_np = image_np.mean(axis=0)

    if image_np.max() <= 1.0:
        return (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(image_np, 0, 255).astype(np.uint8)


def _morph_vessel_mask(img_u8: np.ndarray, threshold: float) -> np.ndarray:
    """CLAHE + black-hat for dark retinal vessels on bright background."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_u8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)
    if blackhat.max() > 0:
        norm = blackhat.astype(np.float32) / float(blackhat.max())
    else:
        norm = blackhat.astype(np.float32)
    return (norm >= threshold).astype(np.float32)


def _frangi_vessel_mask(img_u8: np.ndarray, threshold: float) -> np.ndarray:
    from skimage.filters import frangi

    img_f = img_u8.astype(np.float32) / 255.0
    vesselness = frangi(
        img_f,
        sigmas=range(1, 5),
        black_ridges=True,
    )
    if vesselness.max() > 0:
        vesselness = vesselness / vesselness.max()
    return (vesselness >= threshold).astype(np.float32)


def compute_vessel_mask(
    image_np: np.ndarray,
    backend: str = 'morph',
    threshold: float = 0.25,
    dilate_kernel: int = 3,
) -> np.ndarray:
    """
    Args:
        image_np: (H, W) or (C, H, W), float [0,1] or uint8.
    Returns:
        (H, W) float32 mask in {0, 1}.
    """
    img_u8 = _to_gray_uint8(image_np)
    backend = (backend or 'morph').lower()
    if backend == 'frangi':
        mask = _frangi_vessel_mask(img_u8, threshold)
    else:
        mask = _morph_vessel_mask(img_u8, threshold)

    if dilate_kernel and dilate_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)
        )
        mask = cv2.dilate(mask, k, iterations=1)
    return mask.astype(np.float32)


def compute_vessel_mask_batch(
    images: torch.Tensor,
    backend: str = 'morph',
    threshold: float = 0.25,
    dilate_kernel: int = 3,
) -> torch.Tensor:
    """
    Args:
        images: (B, C, H, W) on any device; values typically in [0, 1].
    Returns:
        (B, 1, H, W) float tensor on the same device as ``images``.
    """
    if images.dim() != 4:
        raise ValueError(f'Expected (B, C, H, W), got {tuple(images.shape)}')

    device = images.device
    dtype = images.dtype
    batch = images.detach().cpu().numpy()
    masks = []
    for i in range(batch.shape[0]):
        masks.append(
            compute_vessel_mask(
                batch[i],
                backend=backend,
                threshold=threshold,
                dilate_kernel=dilate_kernel,
            )
        )
    out = np.stack(masks, axis=0)[:, None, :, :]
    return torch.from_numpy(out).to(device=device, dtype=dtype)
