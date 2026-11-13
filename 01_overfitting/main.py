"""
Практика 1. Недообучение и переобучение своими глазами.

Обучаем три модели разной сложности на одних и тех же данных
и сравниваем ошибку на train и на test.

Запуск (из корня репозитория):  python3 01_overfitting/main.py
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # прячем безвредные технические предупреждения
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# --- 1. Данные -------------------------------------------------------------
# Настоящая закономерность f(x) — плавная волна. Мы её "не знаем",
# видим только точки, к которым добавлен случайный шум (ε из лекции).
np.random.seed(42)


def f(x):
    return np.sin(2 * np.pi * x)


X = np.random.rand(40, 1)                      # 40 объектов, 1 признак
y = f(X[:, 0]) + np.random.normal(0, 0.25, 40)  # целевая переменная = f(X) + шум

# --- 2. Делим на train и test ДО обучения ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
print(f"Объектов в train: {len(X_train)}, в test: {len(X_test)}\n")

# --- 3. Три модели разной сложности --------------------------------------
# degree — степень многочлена. Чем больше, тем более "извилистую"
# кривую может нарисовать модель.
#   1  -> прямая линия
#   4  -> плавная кривая
#   15 -> очень гибкая кривая
degrees = [1, 4, 15]
names = {1: "Слишком простая", 4: "В самый раз", 15: "Слишком сложная"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
x_plot = np.linspace(0, 1, 500).reshape(-1, 1)

print(f"{'Модель':<18} {'Ошибка train':>13} {'Ошибка test':>12}")
print("-" * 45)

for ax, d in zip(axes, degrees):
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X_train, y_train)          # <- обучение: подбор параметров на train

    mse_train = mean_squared_error(y_train, model.predict(X_train))
    mse_test = mean_squared_error(y_test, model.predict(X_test))
    print(f"{names[d]:<18} {mse_train:>13.3f} {mse_test:>12.3f}")

    # Рисуем
    ax.plot(x_plot, f(x_plot), "g--", label="настоящая f(x)")
    ax.plot(x_plot, model.predict(x_plot), "r-", lw=2, label="модель")
    ax.scatter(X_train, y_train, c="tab:blue", label="train")
    ax.scatter(X_test, y_test, c="orange", marker="s", label="test")
    ax.set_ylim(-2, 2)
    ax.set_title(f"{names[d]} (степень {d})\n"
                 f"train: {mse_train:.3f}   test: {mse_test:.3f}")
    ax.legend(loc="lower left", fontsize=8)

# Картинку кладём рядом со скриптом, откуда бы его ни запускали
out_path = Path(__file__).resolve().parent / "overfitting.png"
plt.tight_layout()
plt.savefig(out_path, dpi=110)
print(f"\nГрафик сохранён в {out_path.name}")
