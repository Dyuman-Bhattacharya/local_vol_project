# tests/unit/test_utils_numerical.py
import numpy as np
import pytest

from utils.numerical import (
    finite_diff_first,
    finite_diff_second,
    solve_tridiagonal,
    NumericalError,
)


def test_finite_diff_first_linear():
    # f(x) = 3x + 2 → derivative = 3
    x = np.linspace(0.0, 10.0, 101)
    f = 3.0 * x + 2.0

    df = finite_diff_first(f, x)

    assert np.allclose(df, 3.0, atol=1e-12)


def test_finite_diff_first_quadratic():
    # f(x) = x^2 → f' = 2x
    x = np.linspace(-2.0, 2.0, 201)
    f = x**2

    df = finite_diff_first(f, x)

    assert np.allclose(df[1:-1], 2.0 * x[1:-1], atol=1e-3)


def test_finite_diff_second_quadratic():
    # f(x) = x^2 → f'' = 2
    x = np.linspace(-5.0, 5.0, 501)
    f = x**2

    d2 = finite_diff_second(f, x)

    assert np.allclose(d2[1:-1], 2.0, atol=1e-3)


def test_finite_diff_second_cubic():
    # f(x) = x^3 → f'' = 6x
    x = np.linspace(-2.0, 2.0, 401)
    f = x**3

    d2 = finite_diff_second(f, x)

    assert np.allclose(d2[1:-1], 6.0 * x[1:-1], atol=1e-2)


def test_solve_tridiagonal_identity():
    # Identity matrix → solution equals RHS
    n = 10
    a = np.zeros(n - 1)
    b = np.ones(n)
    c = np.zeros(n - 1)
    d = np.random.randn(n)

    x = solve_tridiagonal(a, b, c, d)

    assert np.allclose(x, d)

def test_solve_tridiagonal_known_system():
    # A = tridiag(-1, 2, -1)
    n = 50
    a = -np.ones(n - 1)
    b = 2.0 * np.ones(n)
    c = -np.ones(n - 1)

    # Exact solution
    i = np.arange(1, n + 1)
    x_exact = i * (n + 1 - i) / (n + 1)

    # Correct RHS for this solution
    d = (2.0 / (n + 1)) * np.ones(n)

    x = solve_tridiagonal(a, b, c, d)

    assert np.allclose(x, x_exact, atol=1e-12)



def test_solve_tridiagonal_zero_pivot_raises():
    a = np.array([1.0])
    b = np.array([0.0, 1.0])  # zero pivot
    c = np.array([1.0])
    d = np.array([1.0, 1.0])

    with pytest.raises(NumericalError):
        solve_tridiagonal(a, b, c, d, check_diagonal=True)
