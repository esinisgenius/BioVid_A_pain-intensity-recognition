from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.thiam2021_signals import preprocess_trial
from src.models.thiam2021_ddcae import ThiamSupervisedDDCAE, joint_loss
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "thiam2021_smoke.yaml").read_text())


def test_phase3_outer_folds_are_subject_isolated_and_complete() -> None:
    samples = pd.read_csv(ROOT / "starting_point" / "samples.csv", sep="\t")
    folds = make_fold_indices(samples, load_phase3_folds(ROOT / CONFIG["data"]["fold_assignments"]))
    assert len(folds) == 5
    seen = []
    for split in folds:
        train_subjects = set(samples.iloc[split["train_index"]]["subject_id"])
        test_subjects = set(samples.iloc[split["test_index"]]["subject_id"])
        assert train_subjects.isdisjoint(test_subjects)
        seen.extend(split["test_index"].tolist())
    assert sorted(seen) == list(range(len(samples)))


def test_signal_preprocessing_produces_three_finite_paper_length_inputs() -> None:
    t = np.arange(2816) / 512.0
    frame = pd.DataFrame(
        {
            "gsr": 2.0 + 0.2 * np.sin(2 * np.pi * 0.1 * t),
            "ecg": np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 40 * t),
            "emg_trapezius": np.sin(2 * np.pi * 80 * t),
        }
    )
    result = preprocess_trial(frame, CONFIG)
    assert set(result) == {"eda", "ecg", "emg"}
    for signal in result.values():
        assert signal.shape == (1152,)
        assert signal.dtype == np.float32
        assert np.isfinite(signal).all()


def test_model_output_reconstruction_and_probability_shapes_are_valid() -> None:
    torch.manual_seed(1)
    model = ThiamSupervisedDDCAE(CONFIG)
    inputs = {name: torch.randn(2, 1, 1152) for name in CONFIG["data"]["modalities"]}
    targets = torch.tensor([0, 4])
    outputs = model(inputs, corrupt=False)
    assert outputs["logits"].shape == (2, 5)
    for name in CONFIG["data"]["modalities"]:
        assert outputs["reconstructions"][name].shape == (2, 1, 1152)
    probabilities = torch.softmax(outputs["logits"], dim=1)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
    loss, pieces = joint_loss(model, outputs, inputs, targets, CONFIG)
    assert torch.isfinite(loss)
    assert set(pieces) >= {"classification", "total", "reconstruction_eda", "reconstruction_ecg", "reconstruction_emg"}
