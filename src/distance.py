"""Hierarchie trid pro vypocet vzdalenosti mezi vektory priznaku.

Zkopírujte sem své řešení z Cvičení 01 — kNN hledá sousedy přes `calculate()`.

Klasifikator k nejblizsich sousedu (``KNNClassifier.predict``) v tomto cviceni
vola metodu ``calculate()`` teto hierarchie po jednotlivych dvojicich bodu:
pro kazdy dotazovany vzorek spocte jeho vzdalenost ke vsem trenovacim bodum,
vybere ``k`` nejblizsich a hlasuje. Bez doplneni ``calculate()`` (a ``is_metric``)
se cviceni nespusti.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    """Abstraktní základní třída pro výpočet vzdálenosti."""

    @property
    @abstractmethod
    def is_metric(self) -> bool:
        """Vrátí True, pokud vzdálenost splňuje axiomy metriky."""

    @abstractmethod
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Vypočítá vzdálenost mezi dvěma 1D vektory."""
        raise NotImplementedError

    def create_distance_matrix(self, x: np.ndarray) -> np.ndarray:
        """Vytvoří symetrickou matici vzdáleností."""
        n_samples = x.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples), dtype=float)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = self.calculate(x[i], x[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix


class EuclideanDistance(Distance):
    """Eukleidovská (L2) vzdálenost."""

    @property
    def is_metric(self) -> bool:
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.ndim == 1 and y.ndim == 1
        assert x.shape == y.shape

        return float(np.linalg.norm(x - y))


class ManhattanDistance(Distance):
    """Manhattanská (L1) vzdálenost."""

    @property
    def is_metric(self) -> bool:
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.ndim == 1 and y.ndim == 1
        assert x.shape == y.shape

        return float(np.sum(np.abs(x - y)))


class CosineCoeficient(Distance):
    """Kosinová vzdálenost (1 - kosinová podobnost)."""

    @property
    def is_metric(self) -> bool:
        return False

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.ndim == 1 and y.ndim == 1
        assert x.shape == y.shape
        assert np.linalg.norm(x) > 0, "x nesmí být nulový vektor"
        assert np.linalg.norm(y) > 0, "y nesmí být nulový vektor"

        similarity = np.dot(x, y) / (
            np.linalg.norm(x) * np.linalg.norm(y)
        )

        similarity = np.clip(similarity, -1.0, 1.0)

        return float(1.0 - similarity)