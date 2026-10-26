"""Strict public fuRo XYZINormal schema over the shared, lossless PCD reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from usim_env_pipeline.models import SourceMetadata
from usim_env_pipeline.pcd import DTYPE as DTYPE, FIELDS as FIELDS
from usim_env_pipeline.pcd import normals as normals, sha256 as sha256, xyz as xyz
from usim_env_pipeline.pcd import open_records as read_records
from usim_env_pipeline.pcd import require_xyzinormal

from usim_env_vtc.sources import PUBLIC_HASHES as PUBLIC_HASHES


def open_records(path: Path) -> tuple[NDArray[np.void], SourceMetadata]:
    records, metadata = read_records(path)
    require_xyzinormal(records, metadata)
    return records, metadata
