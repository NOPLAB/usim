"""Source-only orthographic images for independently annotated edge regions."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


def projection(
    xyz: NDArray[np.float64],
    axes: tuple[int, int],
    path: Path,
    label: str,
    *,
    cell_size: float | None = None,
) -> None:
    """Display measured point density, without fitted edges or repaired surfaces."""
    low = np.quantile(xyz[:, axes], 0.001, axis=0)
    high = np.maximum(np.quantile(xyz[:, axes], 0.999, axis=0), low + 0.1)
    bins: int | list[int] = 768
    if cell_size is not None:
        bins = np.minimum(
            768, np.maximum(1, np.ceil((high - low) / cell_size).astype(int))
        ).tolist()
    density, _, _ = np.histogram2d(
        xyz[:, axes[0]], xyz[:, axes[1]], bins=bins, range=list(zip(low, high, strict=True))
    )
    intensity = np.zeros_like(density, dtype=np.uint8)
    occupied = density > 0
    intensity[occupied] = (
        100 + 155 * np.log1p(density[occupied]) / np.log1p(density.max())
    ).astype(np.uint8)
    image = Image.fromarray(intensity.T[::-1]).convert('RGB')
    image = image.resize((768, 768), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 768, 42), fill=(20, 20, 20))
    draw.text((8, 5), label, fill='white')
    draw.text((8, 23), f'axes {axes}; low {low.round(2)}; high {high.round(2)}', fill='white')
    image.save(path)
