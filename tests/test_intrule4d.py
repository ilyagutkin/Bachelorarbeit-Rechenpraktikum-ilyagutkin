"""
Tests for 4D simplex (pentatope) quadrature rules.
We check polynomial exactness by integrating monomials up to the claimed degree.
"""

import numpy as np
import pytest
import math
from methodsnm.intrule_4d import IntRulePentatope


# -------------------------------------------------------------------------
#  Exaktes Integral eines Monoms über den 4D-Standard-Simplex:
#
#  Δ4 = { x0, x1, x2, x3 >= 0, x0 + x1 + x2 + x3 <= 1 }
#
#  ∫ x0^i x1^j x2^k x3^l dx  =  ( i! j! k! l! ) / ( 4 + i+j+k+l )!
# -------------------------------------------------------------------------
def monomial_integral_pentatope(i, j, k, l):
    deg = i + j + k + l
    num = (
        math.factorial(i)
        * math.factorial(j)
        * math.factorial(k)
        * math.factorial(l)
    )
    den = math.factorial(4 + deg)
    return num / den


# -------------------------------------------------------------------------
# Teste für Quadraturordnung 1, 2, 3, 4, 5:
#  – rule.exactness_degree gibt den garantierten Polynomgrad an.
#  – Wir testen ALLE Monome mit i+j+k+l ≤ deg.
#  – Wenn das hält, stimmt die Quadratur definitiv.
# -------------------------------------------------------------------------
@pytest.mark.parametrize("order", [1,2,3,4,5])
def test_pentatope_monomials_up_to_exactness(order):
    rule = IntRulePentatope(order)
    deg = rule.exactness_degree

    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            for k in range(deg + 1 - i - j):
                for l in range(deg + 1 - i - j - k):

                    # Gesamtgrad prüfen
                    if i + j + k + l > deg:
                        continue

                    # Monomfunktion
                    def f(x, i=i, j=j, k=k, l=l):
                        return (
                            x[0] ** i
                            * x[1] ** j
                            * x[2] ** k
                            * x[3] ** l
                        )

                    # Numerische Quadratur
                    approx = np.sum(
                        rule.weights * np.array([f(p) for p in rule.nodes])
                    )

                    # Exaktes Integral
                    exact = monomial_integral_pentatope(i, j, k, l)

                    # Test
                    assert np.isclose(approx, exact, atol=1e-11), \
                        f"Monom x0^{i} x1^{j} x2^{k} x3^{l} " \
                        f"deg={i+j+k+l} failed for order={order}"

