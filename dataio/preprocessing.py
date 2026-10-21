"""Predzpracovani dat pro cviceni 06.

Obsah:

- ``standardize_fit`` / ``standardize_apply`` (UKOL) -- z-skorova
  standardizace rozdelena na dve faze tak, aby parametry sly spocitat
  **jen z trenovaci mnoziny** (spravny vzor; kontrast s unikem dat).
- ``subsample_imbalance`` (UKOL) -- umele podvzorkovani majoritni tridy
  pro demonstraci, proc pri nevyvazenych datech klame presnost (accuracy).
- ``demonstrate_leakage`` (PREDVYPLNENO) -- spusti obe poradi
  (standardizace pred vs. po rozdeleni) a vrati obe presnosti vedle sebe.
  Vola studentske ``standardize_fit`` / ``standardize_apply``, takze bez
  jejich implementace nedobehne.

Unik dat se demonstruje **spustenim**, ne programovanim, proto zustava
``demonstrate_leakage`` predvyplnena. Pro kNN se v ni pouziva referencni
``sklearn.neighbors.KNeighborsClassifier`` -- jde o ukazku spravne prace
s daty, jadro kNN si student implementuje v ``src/knn.py``.
"""


from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def standardize_fit(x_train: np.ndarray) -> dict:
    """Vypočítá průměr a populační směrodatnou odchylku z trénovacích dat."""
    x_train = np.asarray(x_train, dtype=np.float64)

    assert x_train.ndim == 2, "x_train musí být dvourozměrná matice."
    assert x_train.shape[0] > 0, "Trénovací množina nesmí být prázdná."

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0, ddof=0)

    return {
        "mean": mean,
        "std": std,
    }


def standardize_apply(x: np.ndarray, params: dict) -> np.ndarray:
    """Aplikuje standardizaci podle parametrů naučených na trénovacích datech."""
    assert isinstance(params, dict), "params musí být slovník."
    assert "mean" in params and "std" in params, (
        "params musí obsahovat klíče 'mean' a 'std'."
    )

    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(params["mean"], dtype=np.float64)
    std = np.asarray(params["std"], dtype=np.float64)

    assert x.ndim == 2, "x musí být dvourozměrná matice."
    assert mean.ndim == 1 and std.ndim == 1
    assert x.shape[1] == len(mean), (
        "Počet příznaků neodpovídá parametrům standardizace."
    )
    assert mean.shape == std.shape

    std_safe = np.where(std == 0.0, 1.0, std)

    return np.asarray((x - mean) / std_safe, dtype=np.float64)


def subsample_imbalance(
    x: np.ndarray,
    y: np.ndarray,
    minority_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vytvoří nevyváženou množinu náhodným podvzorkováním jedné třídy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)

    assert 0.0 < minority_ratio < 0.5, (
        "minority_ratio musí být mezi 0 a 0.5."
    )
    assert x.ndim == 2, "x musí být dvourozměrná matice."
    assert y.ndim == 1, "y musí být jednorozměrné pole."
    assert len(x) == len(y), "x a y musí mít stejný počet vzorků."
    assert set(np.unique(y).tolist()) == {0, 1}, (
        "y musí obsahovat obě třídy 0 a 1."
    )

    rng = np.random.default_rng(random_state)

    counts = np.bincount(y.astype(int), minlength=2)
    minority_class = int(np.argmin(counts))
    majority_class = 1 - minority_class

    idx_min = np.flatnonzero(y == minority_class)
    idx_maj = np.flatnonzero(y == majority_class)

    n_min = len(idx_min)
    n_maj = len(idx_maj)

    current_ratio = n_min / (n_min + n_maj)

    if current_ratio > minority_ratio:
        # Ponecháme všechny majoritní vzorky a omezíme minoritní.
        n_keep_min = round(
            minority_ratio * n_maj / (1.0 - minority_ratio)
        )
        n_keep_min = int(np.clip(n_keep_min, 1, n_min))

        keep_min = rng.choice(
            idx_min, size=n_keep_min, replace=False
        )
        keep = np.concatenate([keep_min, idx_maj])

    else:
        # Ponecháme všechny minoritní vzorky a omezíme majoritní.
        n_keep_maj = round(
            n_min * (1.0 - minority_ratio) / minority_ratio
        )
        n_keep_maj = int(np.clip(n_keep_maj, 1, n_maj))

        keep_maj = rng.choice(
            idx_maj, size=n_keep_maj, replace=False
        )
        keep = np.concatenate([idx_min, keep_maj])

    rng.shuffle(keep)

    return x[keep], y[keep]


def demonstrate_leakage(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    random_state: int,
) -> dict:
    """Porovná standardizaci před rozdělením a po rozdělení dat."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)

    test_size = 0.2

    # ŠPATNĚ: standardizace využívá také testovací data.
    params_all = standardize_fit(x)
    x_all_std = standardize_apply(x, params_all)

    xtr_w, xte_w, ytr_w, yte_w = train_test_split(
        x_all_std,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    clf_w = KNeighborsClassifier(n_neighbors=k)
    clf_w.fit(xtr_w, ytr_w)
    acc_wrong = float(np.mean(clf_w.predict(xte_w) == yte_w))

    # SPRÁVNĚ: parametry standardizace se učí pouze z trénovacích dat.
    xtr_r, xte_r, ytr_r, yte_r = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    params_tr = standardize_fit(xtr_r)
    xtr_r_std = standardize_apply(xtr_r, params_tr)
    xte_r_std = standardize_apply(xte_r, params_tr)

    clf_r = KNeighborsClassifier(n_neighbors=k)
    clf_r.fit(xtr_r_std, ytr_r)
    acc_right = float(np.mean(clf_r.predict(xte_r_std) == yte_r))

    return {
        "acc_wrong": acc_wrong,
        "acc_right": acc_right,
        "test_size": test_size,
        "k": k,
    }
