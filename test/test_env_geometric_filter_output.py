"""End-to-end geometric publication preserves source attributes and stage partitions."""

import json
from pathlib import Path

import numpy as np
import pytest

from test.env_pipeline_fixtures import plane_records, write_pcd
from usim_env_pipeline.models import Source
from usim_env_pipeline.point_filter_output import filter_source


def test_geometric_publication_keeps_planar_source_and_original_ids(tmp_path: Path):
    # Given a real archived source with normal, intensity and curvature attributes.
    _ = pytest.importorskip('scipy')
    _ = pytest.importorskip('pxr')
    from usim_env_pipeline.geometric_filter_output import publish_geometric_filter

    records = plane_records()
    path, baseline, output = tmp_path / 'plane.pcd', tmp_path / 'baseline', tmp_path / 'geometry'
    original = write_pcd(path, records)
    _ = filter_source(Source(path), baseline)
    baseline_report = (baseline / 'report.json').read_bytes()
    # When running both real stages and authoring a real point-only USD comparison.
    stage_path = publish_geometric_filter(baseline, output)
    # Then geometry never rejects supported planar points and every source row remains addressable.
    assert stage_path.exists()
    np.testing.assert_array_equal(np.load(output / 'records.npy'), records)
    with np.load(output / 'decisions.npz') as decisions:
        np.testing.assert_array_equal(decisions['ids'], np.arange(len(records)))
        assert not decisions['geometry_rejected'].any()
        np.testing.assert_array_equal(
            decisions['keep'], decisions['sor_keep'] & ~decisions['geometry_rejected']
        )
    report = json.loads((output / 'report.json').read_text())
    assert report['config']['std_ratio'] == 4.0
    assert report['geometry']['geometry_removed_records'] == 0
    assert report['geometry']['sor_retained_records'] == report['retained_records']
    assert (output / 'source' / path.name).read_bytes() == original
    assert (baseline / 'report.json').read_bytes() == baseline_report
