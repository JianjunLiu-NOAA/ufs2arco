import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import xarray as xr

from ufs2arco.sources import GFSArchive

@pytest.fixture
def gfs_dataset():
    t0 = {"start": "2025-01-01T00", "end": "2025-01-01T18", "freq": "6h"}
    fhr = {"start": 0, "end": 6, "step": 6}
    resolution = '0p25'
    chunks = {"t0": 1, "fhr": 1,  "latitude": -1, "longitude": -1}
    return GFSArchive(t0, fhr, resolution, variables=["t", "t2m", "prmsl"], levels=[100, 500, 1000])

def test_init(gfs_dataset):
    assert len(gfs_dataset.t0) == 4
    assert np.array_equal(gfs_dataset.fhr, np.array([0, 6]))

def test_str(gfs_dataset):
    str(gfs_dataset) # just make sure this can run without bugs

def test_name(gfs_dataset):
    assert gfs_dataset.name == "GFSArchive"

def test_open_sample_dataset(gfs_dataset):
    result = gfs_dataset.open_sample_dataset(
        dims={
            "t0": pd.Timestamp("2025-01-01T00"),
            "fhr": 0,
        },
        open_static_vars=True,
        cache_dir="/tmp/cache",
    )
    assert isinstance(result, xr.Dataset)
