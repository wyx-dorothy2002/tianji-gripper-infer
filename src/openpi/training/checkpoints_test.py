import numpy as np
import pytest

from openpi.shared import normalize
from openpi.training import checkpoints


def _norm_stats() -> dict[str, normalize.NormStats]:
    return {
        "state": normalize.NormStats(
            mean=np.array([1.0]),
            std=np.array([2.0]),
            q01=np.array([0.0]),
            q99=np.array([3.0]),
        )
    }


def test_load_norm_stats_uses_matching_asset_id(tmp_path):
    expected = _norm_stats()
    normalize.save(tmp_path / "matching_asset", expected)
    normalize.save(tmp_path / "other_asset", expected)

    actual = checkpoints.load_norm_stats(tmp_path, "matching_asset")

    np.testing.assert_array_equal(actual["state"].mean, expected["state"].mean)


def test_load_norm_stats_falls_back_to_only_checkpoint_stats(tmp_path):
    expected = _norm_stats()
    normalize.save(tmp_path / "checkpoint_asset", expected)

    actual = checkpoints.load_norm_stats(tmp_path, "configured_asset")

    np.testing.assert_array_equal(actual["state"].mean, expected["state"].mean)


def test_load_norm_stats_rejects_ambiguous_fallback(tmp_path):
    expected = _norm_stats()
    normalize.save(tmp_path / "first_asset", expected)
    normalize.save(tmp_path / "second_asset", expected)

    with pytest.raises(FileNotFoundError, match="multiple fallback files"):
        checkpoints.load_norm_stats(tmp_path, "configured_asset")
