"""HPO search space includes expanded LightGBM knobs."""

from __future__ import annotations

from happy_housing.config import ModelConfig
from happy_housing.models.train import _HPO_SPACE, _hpo_candidates


def test_hpo_space_covers_deeper_trees_and_objectives() -> None:
    assert max(_HPO_SPACE["n_estimators"]) >= 2000
    assert min(_HPO_SPACE["learning_rate"]) <= 0.01
    assert "reg_alpha" in _HPO_SPACE
    assert "max_depth" in _HPO_SPACE
    assert "objective" in _HPO_SPACE
    assert "mae" in _HPO_SPACE["objective"]


def test_hpo_candidates_include_yaml_default_and_samples() -> None:
    cfg = ModelConfig(n_estimators=200, learning_rate=0.05, num_leaves=31)
    cands = _hpo_candidates(cfg, trials=12, random_state=0)
    assert len(cands) >= 8
    assert cands[0]["n_estimators"] == 200
    keys = set(cands[1])
    assert {"objective", "reg_alpha", "max_depth"} <= keys
