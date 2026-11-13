"""
Практика 4. Деревья решений и ансамбли (случайный лес, градиентный бустинг)
на нелинейной задаче: механическая скорость проходки (ROP).

В реальном бурении ROP растёт с нагрузкой на долото (WOB) только до
"точки перегиба" (founder point): дальше долото не успевает очищаться,
и рост останавливается. Положение перегиба зависит от расхода промывки.
Такую зависимость прямая линия не опишет. Сравниваем:
линейную регрессию, одиночное дерево (глубокое и обрезанное),
случайный лес (бэггинг) и градиентный бустинг, а затем смотрим,
какие признаки модели считают важными.

Данные СИНТЕТИЧЕСКИЕ: формула ROP придумана для учебных целей
(степенная зависимость от WOB и RPM, насыщение по расходу, перегиб по WOB) + шум.

Запуск (из корня репозитория):  python3 04_trees_ensembles/main.py
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # прячем безвредные технические предупреждения
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error

rng = np.random.default_rng(1)

# --- 1. Синтетические данные ---------------------------------------------
n = 800
wob = rng.uniform(20, 200, n)        # нагрузка на долото, кН
rpm = rng.uniform(60, 200, n)        # частота вращения, об/мин
flow = rng.uniform(15, 45, n)        # расход бурового раствора, л/с
ucs = rng.uniform(30, 120, n)        # прочность породы на сжатие, МПа
mud_temp = rng.uniform(25, 60, n)    # температура раствора на выходе, °C (на ROP не влияет)
pit_vol = rng.uniform(40, 80, n)     # объём в ёмкостях, м3 (на ROP не влияет)


def true_rop(wob, rpm, flow, ucs):
    """Учебная "физика" ROP, м/ч."""
    # Точка перегиба по WOB: чем лучше промывка, тем позже наступает
    wob_founder = 60 + 2.5 * flow
    wob_eff = np.minimum(wob, wob_founder)                 # выше перегиба рост прекращается
    hydraulics = 1 - np.exp(-flow / 15)                    # насыщение по расходу
    return 0.35 * wob_eff ** 0.9 * (rpm / 100) ** 0.5 * hydraulics * (60 / ucs) ** 0.8


rop = true_rop(wob, rpm, flow, ucs) * rng.lognormal(0, 0.10, n)  # шум ~10 %

X = np.column_stack([wob, rpm, flow, ucs, mud_temp, pit_vol])
feature_names = ["WOB, кН", "RPM, об/мин", "Расход, л/с", "UCS, МПа",
                 "T раствора, °C", "Объём ёмкостей, м3"]
print(f"Объектов: {n}, признаков: {X.shape[1]}")
print(f"ROP: от {rop.min():.1f} до {rop.max():.1f} м/ч\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, rop, test_size=0.3, random_state=0
)

# --- 2. Модели -------------------------------------------------------------
# Деревьям масштабирование признаков НЕ нужно: они сравнивают признак
# с порогом, единицы измерения на это не влияют.
models = {
    "Линейная регрессия": LinearRegression(),
    "Дерево (без ограничений)": DecisionTreeRegressor(random_state=0),
    "Дерево (max_depth=4)": DecisionTreeRegressor(max_depth=4, random_state=0),
    "Случайный лес": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                           n_jobs=-1, random_state=0),
    "Градиентный бустинг": GradientBoostingRegressor(n_estimators=300,
                                                     learning_rate=0.05,
                                                     max_depth=3, random_state=0),
}

print(f"{'Модель':<26} {'R2 train':>9} {'R2 test':>8} {'MAE test, м/ч':>14}")
print("-" * 60)
r2_tr, r2_te = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    r2_tr[name] = r2_score(y_train, model.predict(X_train))
    r2_te[name] = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"{name:<26} {r2_tr[name]:>9.3f} {r2_te[name]:>8.3f} {mae:>14.2f}")

# --- 3. Важность признаков -----------------------------------------------
# Перестановочная важность на test: насколько падает качество, если
# перемешать значения одного признака. Честнее встроенной (impurity) важности.
rf = models["Случайный лес"]
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0)
order = np.argsort(perm.importances_mean)
print("\nВажность признаков (случайный лес, падение R2 при перемешивании):")
for i in order[::-1]:
    print(f"  {feature_names[i]:<20} {perm.importances_mean[i]:.3f}")

# --- 4. Картинка -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# (а) R2 на train и test
ax = axes[0]
idx = np.arange(len(models))
ax.barh(idx + 0.2, [r2_tr[m] for m in models], height=0.4,
        color="tab:blue", label="train")
ax.barh(idx - 0.2, [r2_te[m] for m in models], height=0.4,
        color="tab:orange", label="test")
for i, m in enumerate(models):
    ax.text(r2_te[m] + 0.01, i - 0.2, f"{r2_te[m]:.2f}", va="center", fontsize=8)
ax.set_yticks(idx)
ax.set_yticklabels(list(models))
ax.invert_yaxis()
ax.set_xlim(0, 1.3)
ax.set_xlabel("R²")
ax.set_title("Качество моделей\n(разрыв train/test = переобучение)")
ax.legend(loc="upper right")

# (б) ROP от WOB при фиксированных остальных параметрах
ax = axes[1]
wob_grid = np.linspace(20, 200, 300)
fixed = dict(rpm=120, flow=25, ucs=60)
X_grid = np.column_stack([wob_grid,
                          np.full_like(wob_grid, fixed["rpm"]),
                          np.full_like(wob_grid, fixed["flow"]),
                          np.full_like(wob_grid, fixed["ucs"]),
                          np.full_like(wob_grid, 40),
                          np.full_like(wob_grid, 60)])
ax.plot(wob_grid, true_rop(wob_grid, **fixed), "k--", lw=2,
        label="истинная зависимость")
for name, color in [("Линейная регрессия", "gray"),
                    ("Дерево (max_depth=4)", "tab:green"),
                    ("Случайный лес", "tab:red"),
                    ("Градиентный бустинг", "tab:purple")]:
    ax.plot(wob_grid, models[name].predict(X_grid), color=color, lw=1.8,
            label=name)
ax.set_xlabel("WOB, кН")
ax.set_ylabel("ROP, м/ч")
ax.set_title("Срез модели: ROP(WOB)\nRPM=120, расход=25 л/с, UCS=60 МПа")
ax.legend(fontsize=8, loc="lower right")

# (в) перестановочная важность
ax = axes[2]
ax.barh(np.arange(len(order)), perm.importances_mean[order],
        xerr=perm.importances_std[order], color="tab:red", alpha=0.8)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels([feature_names[i] for i in order])
ax.set_xlabel("падение R² при перемешивании признака")
ax.set_title("Важность признаков (случайный лес)\nдва нижних признака на ROP не влияют")

fig.suptitle("ROP: деревья и ансамбли на нелинейной задаче (синтетические данные)",
             fontsize=12)
out_path = Path(__file__).resolve().parent / "trees_ensembles.png"
plt.tight_layout()
plt.savefig(out_path, dpi=110)
print(f"\nГрафик сохранён в {out_path.name}")
