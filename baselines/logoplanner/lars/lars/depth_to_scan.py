"""D435i depth image to pseudo-LiDAR scan conversion."""

import numpy as np
from typing import Tuple, Optional


class DepthToScan:
    """Convert D435i depth image to a 1D pseudo-LiDAR scan.

    Selects a horizontal band from the depth image, divides it into angular
    bins, and takes the minimum valid depth in each bin. Outputs both raw
    (meter) and normalized [0, 1] scans.

    Args:
        scan_dim: Number of angular bins (32, 64, or 128).
        band_rows: (row_start, row_end) in the depth image.
        min_depth: Minimum valid depth in meters.
        max_depth: Maximum valid depth in meters; invalid bins get this.
        hfov_rad: Horizontal field of view in radians.
        input_unit: Unit of input depth: 'mm' or 'meter'.
        normalize: If True, also output a [0, 1] normalized scan.
    """

    def __init__(
        self,
        scan_dim: int = 64,
        band_rows: Tuple[int, int] = (220, 260),
        min_depth: float = 0.1,
        max_depth: float = 8.0,
        hfov_rad: float = 1.518,
        input_unit: str = "meter",
        normalize: bool = True,
    ):
        self.scan_dim = scan_dim
        self.band_rows = band_rows
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hfov_rad = hfov_rad
        self.input_unit = input_unit
        self.normalize = normalize

    def __call__(self, depth_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a depth image to pseudo-LiDAR scan.

        Args:
            depth_image: 2D numpy array (H, W), units per input_unit.

        Returns:
            scan_meter: (scan_dim,) array in meters.
            scan_norm: (scan_dim,) array normalized to [0, 1], or same as
                       scan_meter if normalize=False.
        """
        if self.input_unit == "mm":
            depth = depth_image.astype(np.float32) / 1000.0
        else:
            depth = depth_image.astype(np.float32)

        # Select horizontal band
        r1, r2 = self.band_rows
        band = depth[r1:r2, :]  # (band_height, W)

        # Replace 0, NaN, inf with NaN for per-bin min
        mask = (band > self.min_depth) & (band < self.max_depth) & np.isfinite(band)
        band_clean = np.where(mask, band, np.nan)

        # Split into angular bins
        W = band.shape[1]
        scan = np.full(self.scan_dim, self.max_depth, dtype=np.float32)

        for i in range(self.scan_dim):
            x_start = int(W * i / self.scan_dim)
            x_end = int(W * (i + 1) / self.scan_dim)
            bin_vals = band_clean[:, x_start:x_end]
            bin_min = np.nanmin(bin_vals)
            if np.isfinite(bin_min):
                scan[i] = np.clip(bin_min, self.min_depth, self.max_depth)

        scan_meter = scan.astype(np.float32)

        if self.normalize:
            scan_norm = (scan_meter - self.min_depth) / (self.max_depth - self.min_depth)
            scan_norm = np.clip(scan_norm, 0.0, 1.0).astype(np.float32)
        else:
            scan_norm = scan_meter.copy()

        return scan_meter, scan_norm


def create_depth_to_scan_from_config(config: dict) -> DepthToScan:
    """Factory to create DepthToScan from a config dictionary."""
    cfg = config.get("depth_to_scan", config)
    return DepthToScan(
        scan_dim=cfg.get("scan_dim", 64),
        band_rows=tuple(cfg.get("band_rows", [220, 260])),
        min_depth=cfg.get("min_depth", 0.1),
        max_depth=cfg.get("max_depth", 8.0),
        hfov_rad=cfg.get("hfov_rad", 1.518),
        input_unit=cfg.get("input_unit", "meter"),
        normalize=cfg.get("normalize", True),
    )
