"""
Практика 5. Кросс-валидация по скважинам (GroupKFold): почему случайное
разбиение даёт слишком оптимистичную оценку.

Задача: по каротажу (ГК, плотность, нейтронный метод) и глубине предсказать
пористость по керну. Данные идут по глубине, соседние точки одной скважины
почти одинаковые. Если делить выборку случайно, в test попадают точки,
у которых в train есть "соседи-близнецы" из той же скважины, — модель
не обобщает, а интерполирует. На практике же модель применяют к НОВОЙ
скважине. Честная проверка — делить по скважинам целиком (GroupKFold).

Данные СИНТЕТИЧЕСКИЕ: 8 скважин, плавные профили пористости и глинистости
по глубине, каротаж получен из них по упрощённым петрофизическим формулам
+ погрешность приборов, своя для каждой скважины.

Запуск (из корня репозитория):  python3 05_group_cv/main.py
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # прячем безвредные технические предупреждения
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

rng = np.random.default_rng(3)

# --- 1. Синтетические данные: 8 скважин ------------------------------------
n_wells = 8
n_points = 150                                   # точек по глубине в скважине
depth = np.linspace(2500, 2650, n_points)        # м, шаг ~1 м


def smooth_curve(scale, n_waves=4):
    """Плавная случайная кривая по глубине (сумма синусоид)."""
    z = (depth - depth[0]) / (depth[-1] - depth[0])
    out = np.zeros_like(z)
    for _ in range(n_waves):
        freq = rng.uniform(0.5, 4)
        phase = rng.uniform(0, 2 * np.pi)
        out += rng.normal(0, 1) * np.sin(2 * np.pi * freq * z + phase)
    return scale * out / n_waves


rows = []
for w in range(n_wells):
    phi = np.clip(0.16 + smooth_curve(0.10), 0.02, 0.32)    # пористость керна
    vsh = np.clip(0.30 + smooth_curve(0.35), 0.0, 1.0)      # глинистость
    # Погрешность плотномера своя в каждой скважине (каверны, калибровка)
    # и плавно меняется по глубине — по самому каротажу её не увидеть.
    rho_bias = rng.normal(0, 0.04) + smooth_curve(0.06)
    rhob = 2.65 - 1.65 * phi + 0.08 * vsh + rho_bias + rng.normal(0, 0.015, n_points)
    nphi = phi + 0.12 * vsh + rng.normal(0, 0.015, n_points)
    gr = 20 + 110 * vsh + rng.normal(0, 5, n_points)
    for i in range(n_points):
        rows.append((w, depth[i], gr[i], rhob[i], nphi[i], phi[i]))

data = np.array(rows)
groups = data[:, 0].astype(int)                  # номер скважины
X = data[:, 1:5]                                 # глубина, ГК, RHOB, NPHI
y = data[:, 5]                                   # пористость по керну
print(f"Скважин: {n_wells}, точек всего: {len(y)}, "
      f"признаки: глубина, ГК, RHOB, NPHI\n")

# --- 2. Две схемы кросс-валидации ------------------------------------------
# KFold(shuffle=True) — точки перемешаны, скважины "размазаны" по фолдам.
# GroupKFold        — каждая скважина целиком либо в train, либо в test.
schemes = {
    "Случайный KFold": (KFold(n_splits=4, shuffle=True, random_state=0), None),
    "GroupKFold по скважинам": (GroupKFold(n_splits=4), groups),
}
models = {
    "Линейная регрессия": LinearRegression(),
    "Случайный лес": RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                           random_state=0),
}

results = {}
print(f"{'Модель':<20} {'Схема CV':<25} {'R2 (среднее ± std)':>20}")
print("-" * 68)
for m_name, model in models.items():
    for s_name, (cv, grp) in schemes.items():
        scores = cross_val_score(model, X, y, cv=cv, groups=grp, scoring="r2")
        results[(m_name, s_name)] = scores
        print(f"{m_name:<20} {s_name:<25} "
              f"{scores.mean():>11.3f} ± {scores.std():.3f}")

gap = (results[("Случайный лес", "Случайный KFold")].mean()
       - results[("Случайный лес", "GroupKFold по скважинам")].mean())
print(f"\nСлучайный лес: случайное разбиение завышает R2 на {gap:.2f}.")
print("Оценка по GroupKFold — то, чего ждать на новой скважине.")

# --- 3. Картинка -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                         gridspec_kw={"width_ratios": [1.3, 1, 1]})

# (а) сравнение оценок качества
ax = axes[0]
x = np.arange(len(models))
for k, (s_name, color) in enumerate(zip(schemes, ["tab:red", "tab:green"])):
    means = [results[(m, s_name)].mean() for m in models]
    stds = [results[(m, s_name)].std() for m in models]
    bars = ax.bar(x + (k - 0.5) * 0.38, means, width=0.38, yerr=stds,
                  capsize=4, color=color, label=s_name)
    for b, val, sd in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, max(val, 0) + sd + 0.03,
                f"{val:.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(list(models))
ax.axhline(0, color="black", lw=0.8)
ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1.15)
ax.set_ylabel("R² на кросс-валидации")
ax.set_title("Одна и та же модель — разные оценки\n"
             "(случайное разбиение завышает качество)")
ax.legend(loc="upper left", fontsize=9)

# (б), (в) какие точки попадают в test на первом фолде
for ax, (s_name, (cv, grp)) in zip(axes[1:], schemes.items()):
    _, test_idx = next(iter(cv.split(X, y, groups=grp)))
    is_test = np.zeros(len(y), dtype=bool)
    is_test[test_idx] = True
    ax.scatter(groups[~is_test] + 1, X[~is_test, 0], s=4, c="tab:blue",
               label="train")
    ax.scatter(groups[is_test] + 1, X[is_test, 0], s=4, c="tab:orange",
               label="test")
    ax.set_ylim(depth[-1] + 30, depth[0] - 5)    # глубина растёт вниз, снизу место под легенду
    ax.set_xticks(range(1, n_wells + 1))
    ax.set_xlabel("номер скважины")
    ax.set_ylabel("глубина, м")
    ax.set_title(f"{s_name}: фолд 1")
    ax.legend(loc="lower center", ncol=2, markerscale=3, fontsize=8)

fig.suptitle("Пористость по ГИС: проверка на новой скважине "
             "(синтетические данные)", fontsize=12)
out_path = Path(__file__).resolve().parent / "group_cv.png"
plt.tight_layout()
plt.savefig(out_path, dpi=110)
print(f"\nГрафик сохранён в {out_path.name}")
