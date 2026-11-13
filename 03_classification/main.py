"""
Практика 3. Классификация и метрики: коллектор или неколлектор по данным ГИС.

Задача: по четырём каротажным кривым (ГК, плотность, нейтронная пористость,
сопротивление) определить, является ли интервал коллектором. Коллекторов
в разрезе мало (~15 %), поэтому одна точность (accuracy) обманчива:
модель "всегда неколлектор" уже даёт ~85 %. Сравниваем логистическую
регрессию и k ближайших соседей по матрице ошибок, precision, recall, F1
и ROC-кривой.

Данные СИНТЕТИЧЕСКИЕ: значения каротажа сгенерированы в коде
(нормальные распределения с правдоподобными для песчаников и глин
средними и сильным перекрытием классов).

Запуск (из корня репозитория):  python3 03_classification/main.py
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # прячем безвредные технические предупреждения
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

rng = np.random.default_rng(7)

# --- 1. Синтетические данные ГИС -------------------------------------------
# Класс 1 = коллектор (нефтенасыщенный песчаник), класс 0 = неколлектор
# (глины, плотные породы). Средние подобраны "по учебнику", разброс большой,
# чтобы классы заметно перекрывались, как в реальном разрезе.
n = 1500
y = (rng.random(n) < 0.15).astype(int)          # ~15 % коллекторов


def make_logs(label):
    """Возвращает (ГК, RHOB, NPHI, log10 RT) для одного объекта."""
    if label == 1:                              # коллектор
        gr = rng.normal(65, 22)                 # гамма-каротаж, API
        rhob = rng.normal(2.36, 0.10)           # плотность, г/см3
        nphi = rng.normal(0.22, 0.06)           # нейтронная пористость, д.ед.
        log_rt = rng.normal(1.0, 0.45)          # lg сопротивления, Ом*м
    else:                                       # неколлектор
        gr = rng.normal(88, 25)
        rhob = rng.normal(2.44, 0.10)
        nphi = rng.normal(0.26, 0.07)
        log_rt = rng.normal(0.65, 0.45)
    return gr, rhob, nphi, log_rt


X = np.array([make_logs(lbl) for lbl in y])
feature_names = ["ГК, API", "RHOB, г/см3", "NPHI, д.ед.", "lg RT"]
print(f"Объектов: {n}, из них коллекторов: {y.sum()} ({y.mean():.0%})\n")

# --- 2. Train / test с сохранением доли классов (stratify) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# --- 3. Модели -------------------------------------------------------------
# Масштабирование нужно обеим моделям: логистической регрессии — для
# регуляризации, kNN — потому что он считает расстояния (иначе ГК в
# десятках API "задавит" плотность в долях г/см3).
models = {
    "Всегда 'неколлектор'": DummyClassifier(strategy="most_frequent"),
    "Логистическая регрессия": make_pipeline(StandardScaler(),
                                             LogisticRegression()),
    "kNN, k=15": make_pipeline(StandardScaler(),
                               KNeighborsClassifier(n_neighbors=15)),
}

print(f"{'Модель':<25} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} "
      f"{'F1':>5} {'ROC AUC':>8}")
print("-" * 67)
proba = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)                 # порог 0.5 по умолчанию
    p = model.predict_proba(X_test)[:, 1]        # "уверенность" модели
    proba[name] = p
    print(f"{name:<25} {accuracy_score(y_test, pred):>8.3f} "
          f"{precision_score(y_test, pred, zero_division=0):>9.3f} "
          f"{recall_score(y_test, pred):>7.3f} "
          f"{f1_score(y_test, pred, zero_division=0):>5.3f} "
          f"{roc_auc_score(y_test, p):>8.3f}")

# --- 4. Порог классификации — это инженерное решение ----------------------
# Пропустить коллектор (FN) обычно дороже, чем лишний раз его проверить (FP).
# Снизим порог для логистической регрессии и посмотрим на recall.
logreg_p = proba["Логистическая регрессия"]
print("\nЛогистическая регрессия при разных порогах:")
print(f"{'Порог':>6} {'Precision':>10} {'Recall':>7}")
for thr in [0.5, 0.3, 0.2]:
    pred_thr = (logreg_p >= thr).astype(int)
    print(f"{thr:>6.1f} {precision_score(y_test, pred_thr):>10.3f} "
          f"{recall_score(y_test, pred_thr):>7.3f}")

# --- 5. Картинка -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
labels = ["неколлектор", "коллектор"]

# (а), (б) матрицы ошибок при пороге 0.5 и 0.3
for ax, thr in zip(axes[:2], [0.5, 0.3]):
    cm = confusion_matrix(y_test, (logreg_p >= thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"Логистическая регрессия, порог {thr}\n"
                 f"recall = {tp / (tp + fn):.2f},  "
                 f"precision = {tp / (tp + fp):.2f}")
    ax.set_xlabel("предсказанный класс")
    ax.set_ylabel("истинный класс")

# (в) ROC-кривые
ax = axes[2]
for name, color in zip(models, ["gray", "tab:blue", "tab:orange"]):
    fpr, tpr, _ = roc_curve(y_test, proba[name])
    auc = roc_auc_score(y_test, proba[name])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.2f})")
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_xlabel("FPR — доля ложных тревог")
ax.set_ylabel("TPR (recall) — доля найденных коллекторов")
ax.set_title("ROC-кривые на test")
ax.legend(loc="lower right", fontsize=8)

fig.suptitle("Коллектор / неколлектор по ГИС (синтетические данные)",
             fontsize=12)
out_path = Path(__file__).resolve().parent / "classification.png"
plt.tight_layout()
plt.savefig(out_path, dpi=110)
print(f"\nГрафик сохранён в {out_path.name}")
