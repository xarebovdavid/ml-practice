"""
Дымовой тест: каждая практика запускается без ошибок и сохраняет картинку.

Запуск (из корня репозитория):  python3 -m pytest -q
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# папка практики -> имя PNG, который она должна сохранить
PRACTICES = {
    "01_overfitting": "overfitting.png",
    "02_regularization": "regularization.png",
    "03_classification": "classification.png",
    "04_trees_ensembles": "trees_ensembles.png",
    "05_group_cv": "group_cv.png",
}


@pytest.mark.parametrize("folder, png", PRACTICES.items())
def test_practice_runs_and_saves_png(folder, png):
    script = ROOT / folder / "main.py"
    out_png = ROOT / folder / png
    if out_png.exists():
        out_png.unlink()                      # проверяем, что картинка создаётся заново

    env = dict(os.environ, MPLBACKEND="Agg")  # без окон, только запись в файл
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    assert "График сохранён" in result.stdout
    assert out_png.exists() and out_png.stat().st_size > 10_000
