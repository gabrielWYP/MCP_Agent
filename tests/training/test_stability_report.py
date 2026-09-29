"""scripts/stability_report.py metrics on a synthetic metrics_history.csv."""

import csv

import pytest

from scripts.stability_report import compute_stability, main, parse_run, read_history

# Mango alternates 1.0 / 0.0 (the observed oscillation); damage ramps.
_MANGO = [1.0, 0.0, 1.0, 0.0, 0.95, 0.5]
_DAMAGE = [0.0, 0.1, 0.2, 0.3, 0.2, 0.4]


@pytest.fixture
def history(tmp_path):
    path = tmp_path / "run_a" / "metrics_history.csv"
    path.parent.mkdir()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "map50", "ap50_class_0", "ap50_class_1"])
        for i, (m, d) in enumerate(zip(_MANGO, _DAMAGE), start=1):
            writer.writerow([i, (m + d) / 2, m, d])
    return path


def test_full_window(history):
    stats = compute_stability(read_history(history), "a", last=30)
    assert stats.epochs == 6 and stats.window == 6
    assert stats.mango_mean == pytest.approx(sum(_MANGO) / 6)
    mean = sum(_MANGO) / 6
    assert stats.mango_std == pytest.approx((sum((v - mean) ** 2 for v in _MANGO) / 6) ** 0.5)
    assert stats.mango_mean_abs_delta == pytest.approx((1 + 1 + 1 + 0.95 + 0.45) / 5)
    assert stats.mango_frac_ge_threshold == pytest.approx(3 / 6)
    assert stats.damage_max == pytest.approx(0.4)
    assert stats.damage_mean == pytest.approx(sum(_DAMAGE) / 6)
    assert stats.damage_mean_abs_delta == pytest.approx((0.1 + 0.1 + 0.1 + 0.1 + 0.2) / 5)


def test_last_n_window(history):
    stats = compute_stability(read_history(history), "a", last=2)
    assert stats.window == 2
    assert stats.mango_mean == pytest.approx((0.95 + 0.5) / 2)
    assert stats.mango_mean_abs_delta == pytest.approx(0.45)
    assert stats.damage_max == pytest.approx(0.4)


def test_single_epoch_window_has_zero_delta(history):
    stats = compute_stability(read_history(history), "a", last=1)
    assert stats.mango_mean_abs_delta == 0.0


def test_parse_run_labels():
    assert parse_run("base=x/metrics_history.csv")[0] == "base"
    assert parse_run("runs/ema/metrics_history.csv")[0] == "ema"


def test_cli_writes_markdown(history, tmp_path, capsys):
    out = tmp_path / "report.md"
    assert main([f"osc={history}", "--last", "30", "--out", str(out)]) == 0
    text = out.read_text()
    assert "| osc | 6 | 6 |" in text
    assert text in capsys.readouterr().out


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("epoch,map50\n1,0.5\n")
    with pytest.raises(ValueError, match="ap50_class_0"):
        read_history(path)
