"""
Практика 2. Линейная регрессия и регуляризация (Ridge, Lasso) на примере
механической скорости проходки (ROP).

Задача: по режимным параметрам бурения предсказать ROP. Кроме полезных
параметров в таблицу попали 20 "лишних" каналов станции ГТИ, которые на ROP
не влияют. Данных мало, признаков много -> обычная линейная регрессия
переобучается. Смотрим, как штраф на коэффициенты (регуляризация) это лечит,
а силу штрафа alpha подбираем k-блочной кросс-валидацией.

Данные СИНТЕТИЧЕСКИЕ: сгенерированы в коде по упрощённой модели типа
Бингхэма (ROP ~ (WOB/D)^a * RPM^b / прочность породы) + шум.

Запуск (из корня репозитория):  python3 02_regularization/main.py
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # прячем безвредные технические предупреждения
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)

# --- 1. Синтетические данные ---------------------------------------------
# Режимные параметры (типичные диапазоны для долота 215.9 мм):
n = 120
wob = rng.uniform(40, 180, n)        # нагрузка на долото, кН
rpm = rng.uniform(60, 200, n)        # частота вращения, об/мин
flow = rng.uniform(20, 45, n)        # расход бурового раствора, л/с
ucs = rng.uniform(30, 120, n)        # прочность породы на сжатие, МПа
d_bit = 0.2159                       # диаметр долота, м

# Модель типа Бингхэма: ROP = K * (WOB/D)^a * RPM^b * Q^c / UCS^e
# После логарифмирования она становится ЛИНЕЙНОЙ по log-признакам:
# log ROP = log K + a*log(WOB/D) + b*log RPM + c*log Q - e*log UCS
a, b, c, e = 1.2, 0.6, 0.3, 0.9
log_rop = (np.log(0.01) + a * np.log(wob / d_bit) + b * np.log(rpm)
           + c * np.log(flow) - e * np.log(ucs))
log_rop += rng.normal(0, 0.15, n)    # шум измерений и неучтённые факторы
rop = np.exp(log_rop)                # м/ч

# Полезные признаки (в логарифмах, как подсказывает физика):
useful = np.column_stack([np.log(wob / d_bit), np.log(rpm),
                          np.log(flow), np.log(ucs)])
useful_names = ["log(WOB/D)", "log(RPM)", "log(Q)", "log(UCS)"]

# "Лишние" каналы: случайные сигналы, никак не связанные с ROP
n_noise = 20
noise = rng.normal(0, 1, (n, n_noise))
noise_names = [f"канал_{i + 1}" for i in range(n_noise)]

X = np.hstack([useful, noise])
y = np.log(rop)                      # учим модель на log(ROP)
feature_names = useful_names + noise_names
print(f"Объектов: {n}, признаков: {X.shape[1]} "
      f"(полезных {len(useful_names)}, лишних {n_noise})")
print(f"ROP в выборке: от {rop.min():.1f} до {rop.max():.1f} м/ч\n")

# --- 2. Train / test ------------------------------------------------------
# Специально берём маленький train: так переобучение заметнее
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=30, random_state=0
)

# --- 3. Подбор alpha кросс-валидацией -------------------------------------
# Масштабирование ОБЯЗАТЕЛЬНО: штраф одинаков для всех коэффициентов,
# поэтому признаки должны быть в одних единицах (среднее 0, std 1).
# Pipeline гарантирует, что scaler учится только на train-фолдах.
cv = KFold(n_splits=5, shuffle=True, random_state=0)
alphas = np.logspace(-4, 2, 40)


def cv_mse(model):
    """Средняя MSE на 5 фолдах кросс-валидации (только train!)."""
    scores = cross_val_score(make_pipeline(StandardScaler(), model),
                             X_train, y_train, cv=cv,
                             scoring="neg_mean_squared_error")
    return -scores.mean()


ridge_cv = [cv_mse(Ridge(alpha=al)) for al in alphas]
lasso_cv = [cv_mse(Lasso(alpha=al, max_iter=50_000)) for al in alphas]
ols_cv = cv_mse(LinearRegression())

best_ridge = alphas[int(np.argmin(ridge_cv))]
best_lasso = alphas[int(np.argmin(lasso_cv))]

# --- 4. Финальное сравнение на test ----------------------------------------
models = {
    "Линейная (без штрафа)": LinearRegression(),
    f"Ridge, alpha={best_ridge:.3g}": Ridge(alpha=best_ridge),
    f"Lasso, alpha={best_lasso:.3g}": Lasso(alpha=best_lasso, max_iter=50_000),
}

print(f"{'Модель':<26} {'MSE train':>10} {'MSE test':>9} {'R2 test':>8}")
print("-" * 56)
fitted = {}
for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), model).fit(X_train, y_train)
    fitted[name] = pipe
    mse_tr = mean_squared_error(y_train, pipe.predict(X_train))
    mse_te = mean_squared_error(y_test, pipe.predict(X_test))
    r2_te = r2_score(y_test, pipe.predict(X_test))
    print(f"{name:<26} {mse_tr:>10.4f} {mse_te:>9.4f} {r2_te:>8.3f}")

lasso_coef = list(fitted.values())[2][-1].coef_
kept = [nm for nm, cf in zip(feature_names, lasso_coef) if abs(cf) > 1e-6]
print(f"\nLasso оставил ненулевыми {len(kept)} из {len(feature_names)} "
      f"признаков: {', '.join(kept)}")

# --- 5. Картинка -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (а) ошибка на кросс-валидации в зависимости от силы штрафа
ax = axes[0]
ax.loglog(alphas, ridge_cv, "-", color="tab:blue", lw=2, label="Ridge (L2)")
ax.loglog(alphas, lasso_cv, "-", color="tab:red", lw=2, label="Lasso (L1)")
ax.axhline(ols_cv, color="gray", ls="--", label="без регуляризации")
ax.axvline(best_ridge, color="tab:blue", ls=":", alpha=0.7)
ax.axvline(best_lasso, color="tab:red", ls=":", alpha=0.7)
ax.set_xlabel("сила штрафа alpha (лог. шкала)")
ax.set_ylabel("MSE на 5-блочной кросс-валидации (лог. шкала)")
ax.set_title("Подбор alpha кросс-валидацией\n(пунктир — лучшее alpha для каждой модели)")
ax.legend()

# (б) коэффициенты моделей: какие признаки модели считают важными
ax = axes[1]
idx = np.arange(len(feature_names))
w = 0.27
for k, (name, color) in enumerate(zip(fitted, ["gray", "tab:blue", "tab:red"])):
    ax.bar(idx + (k - 1) * w, fitted[name][-1].coef_, width=w,
           color=color, label=name.split(",")[0])
ax.axvspan(-0.5, len(useful_names) - 0.5, color="tab:green", alpha=0.08)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(idx)
ax.set_xticklabels(feature_names, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("коэффициент (признаки стандартизованы)")
ax.set_title("Коэффициенты моделей\n(зелёная зона — признаки, реально влияющие на ROP)")
ax.legend(fontsize=9)

fig.suptitle("ROP: регуляризация гасит лишние признаки (синтетические данные)",
             fontsize=12)
out_path = Path(__file__).resolve().parent / "regularization.png"
plt.tight_layout()
plt.savefig(out_path, dpi=110)
print(f"\nГрафик сохранён в {out_path.name}")
