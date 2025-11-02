
"""
Wang & Song (2019) benchmark functions f1–f24

Centralized function definitions for reuse across SPHH/GPHH experiments.
All objectives are **minimization**.

Usage
-----
from benchmark_functions import FUNCTIONS, OBJECTIVE, f1  # etc.
y = f1(x)  # numpy array x

Auto-generated from `single_point_sphh.py`.
"""
from __future__ import annotations
import numpy as np


def f1(x: np.ndarray) -> float:
    # x is 2D: [x1, x2]
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]


def f2(x: np.ndarray) -> float:
    # Six-Hump Camel function (2D)
    # f2(x1,x2) = 4x1^2 - 2.1x1^4 + (1/3)x1^6 + x1*x2 - 4x2^2 + 4x2^4
    x1, x2 = x[0], x[1]
    return 4*x1**2 - 2.1*x1**4 + (x1**6)/3.0 + x1*x2 - 4*x2**2 + 4*x2**4


def f3(x: np.ndarray) -> float:
    # Sphere (scalable to any D)
    return float(np.sum(x**2))


def f4(x: np.ndarray) -> float:
    # Schwefel 2.22 (scalable): sum |x_i| + prod |x_i|
    ax = np.abs(x)
    return float(ax.sum() + ax.prod())


def f5(x: np.ndarray) -> float:
    # Schwefel 1.2 (scalable): sum_i (sum_{j<=i} x_j)^2
    c = np.cumsum(x)
    return float(np.sum(c * c))


def f6(x: np.ndarray) -> float:
    # Schwefel 2.21 (scalable): max |x_i|
    return float(np.max(np.abs(x)))


def f7(x: np.ndarray) -> float:
    # Weighted Sphere: sum_{i=1..D} i * x_i^2
    idx = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(idx * (x ** 2)))


def f8(x: np.ndarray) -> float:
    # Weighted Quartic: sum_{i=1..D} i * x_i^4
    idx = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(idx * (x ** 4)))


def f9(x: np.ndarray) -> float:
    # Sum |x_i|^(i+1), i = 1..D
    ax = np.abs(x)
    exps = np.arange(2, x.size + 2, dtype=float)  # i+1
    return float(np.sum(ax ** exps))


def f10(x: np.ndarray) -> float:
    # Elliptic / weighted sphere: sum 10^(6*(i-1)/(D-1)) * x_i^2
    D = x.size
    exps = np.linspace(0.0, 6.0, D)          # 0 ... 6
    weights = 10.0 ** exps                   # 1 ... 1e6
    return float(np.sum(weights * (x ** 2)))


def f11(x: np.ndarray) -> float:
    # Step function: sum floor(x_i + 0.5)^2
    y = np.floor(x + 0.5)
    return float(np.sum(y * y))


def f12(x: np.ndarray) -> float:
    # Quartic with noise: sum_{i=1..D} i * x_i^4 + random[0,1)
    idx = np.arange(1, x.size + 1, dtype=float)
    quartic = np.sum(idx * (x ** 4))
    noise = np.random.rand()
    return float(quartic + noise)


def f13(x: np.ndarray) -> float:
    # Rastrigin (scalable)
    return float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))


def f14(x: np.ndarray) -> float:
    # Ackley (standard form; optimum 0 at x=0)
    D = x.size
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / D))
    term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * x)))
    return float(term1 + term2 + 20.0 + np.e)


def f15(x: np.ndarray) -> float:
    # Griewank: (1/4000) * sum x_i^2 - prod cos(x_i / sqrt(i)) + 1
    idx = np.arange(1, x.size + 1, dtype=float)
    quad = np.sum(x * x) / 4000.0
    trig = np.prod(np.cos(x / np.sqrt(idx)))
    return float(quad - trig + 1.0)


def f16(x: np.ndarray) -> float:
    # Schaffer F6-style: 0.5 + (sin^2(sqrt(sum x^2)) - 0.5) / (1 + 0.001*sum x^2)^2
    s2 = np.sum(x * x)
    num = np.sin(np.sqrt(s2))**2 - 0.5
    den = (1.0 + 0.001 * s2)**2
    return float(0.5 + num / den)


def f17(x: np.ndarray) -> float:
    # Styblinski–Tang (averaged over D): (1/D) * sum (x_i^4 - 16 x_i^2 + 5 x_i)
    return float(np.mean(x**4 - 16.0 * x**2 + 5.0 * x))


def f18(x: np.ndarray) -> float:
    # Non-smooth: sum |x_i * sin(x_i) + 0.1 * x_i|
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))


def f19(x: np.ndarray) -> float:
    # Non-continuous Rastrigin
    # y_i = x_i            if |x_i| < 0.5
    #     = round(2*x_i)/2 otherwise
    y = np.where(np.abs(x) < 0.5, x, np.round(2.0 * x) / 2.0)
    return float(np.sum(y**2 - 10.0 * np.cos(2.0 * np.pi * y) + 10.0))


def f20(x: np.ndarray) -> float:
    # Penalized function (Penalized #1)
    D = x.size
    y = 1.0 + (x + 1.0) / 4.0

    sin2 = np.sin(np.pi * y) ** 2
    term_bracket = (
        10.0 * sin2[0]
        + np.sum((y[:-1] - 1.0) ** 2 * (1.0 + 10.0 * sin2[1:]))
        + (y[-1] - 1.0) ** 2
    )
    main = (np.pi / D) * term_bracket

    # penalty u(x_i, a=10, k=100, m=4)
    a, k, m = 10.0, 100.0, 4.0
    u = np.where(
        x > a, k * (x - a) ** m,
        np.where(x < -a, k * (-x - a) ** m, 0.0)
    )
    return float(main + np.sum(u))


def f21(x: np.ndarray) -> float:
    # Schwefel (2.26): 418.982887...*D - sum(x_i * sin(sqrt(|x_i|)))
    D = x.size
    return float(418.9828872724338 * D - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def f22(x: np.ndarray) -> float:
    # Scaled/anisotropic Rastrigin:
    # sum_i [ (w_i * x_i)^2 - 10*cos(2*pi*w_i*x_i) + 10 ], w_i = 10^linspace(0,1,D)
    D = x.size
    w = 10.0 ** np.linspace(0.0, 1.0, D)
    z = w * x
    return float(np.sum(z*z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))


def f23(x: np.ndarray) -> float:
    # Penalized #2
    D = x.size
    # main bracketed term (note the linkage i -> i+1 and special last term)
    term1 = np.sin(3.0 * np.pi * x[0]) ** 2
    if D > 1:
        middle = np.sum((x[:-1] - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x[1:]) ** 2))
    else:
        middle = 0.0
    term_last = (x[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x[-1]) ** 2)
    main = 0.1 * (term1 + middle + term_last)

    # penalty u(x_i; a=5, k=100, m=4)
    a, k, m = 5.0, 100.0, 4.0
    u_pos = np.where(x > a, k * (x - a) ** m, 0.0)
    u_neg = np.where(x < -a, k * (-x - a) ** m, 0.0)
    penalty = np.sum(u_pos + u_neg)

    return float(main + penalty)


def f24(x: np.ndarray) -> float:
    # Penalized variant with transform ω_i = 1 + (x_i - 1)/4
    w = 1.0 + (x - 1.0) / 4.0
    D = x.size
    term1 = np.sin(np.pi * w[0]) ** 2
    if D > 1:
        middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * (np.sin(np.pi * w[:-1] + 1.0) ** 2)))
    else:
        middle = 0.0
    term_last = (w[-1] - 1.0) ** 2 * (1.0 + (np.sin(2.0 * np.pi * w[-1]) ** 2))
    return float(term1 + middle + term_last)


# =============================================================================
# Assignment-2 objective registry: name -> (func, lo, hi, dim)
# =============================================================================
OBJECTIVES = {
    "f1": (f1, np.array([-10.0, -10.0]), np.array([10.0, 10.0]), 2),
    "f2": (f2, np.array([-5.0,  -5.0 ]), np.array([ 5.0,  5.0 ]), 2),
    "f3_D10": (f3, np.full(10, -100.0), np.full(10, 100.0), 10),
    "f3_D30": (f3, np.full(30, -100.0), np.full(30, 100.0), 30),
    "f3_D50": (f3, np.full(50, -100.0), np.full(50, 100.0), 50),
    "f4_D10": (f4, np.full(10, -10.0), np.full(10,  10.0), 10),
    "f4_D30": (f4, np.full(30, -10.0), np.full(30,  10.0), 30),
    "f4_D50": (f4, np.full(50, -10.0), np.full(50,  10.0), 50),
    "f5_D10": (f5, np.full(10, -100.0), np.full(10, 100.0), 10),
    "f5_D30": (f5, np.full(30, -100.0), np.full(30, 100.0), 30),
    "f5_D50": (f5, np.full(50, -100.0), np.full(50, 100.0), 50),
    "f6_D10": (f6, np.full(10, -100.0), np.full(10, 100.0), 10),
    "f6_D30": (f6, np.full(30, -100.0), np.full(30, 100.0), 30),
    "f6_D50": (f6, np.full(50, -100.0), np.full(50, 100.0), 50),
    "f7_D10": (f7, np.full(10, -10.0), np.full(10,  10.0), 10),
    "f7_D30": (f7, np.full(30, -10.0), np.full(30,  10.0), 30),
    "f7_D50": (f7, np.full(50, -10.0), np.full(50,  10.0), 50),
    "f8_D10": (f8, np.full(10, -1.28), np.full(10,  1.28), 10),
    "f8_D30": (f8, np.full(30, -1.28), np.full(30,  1.28), 30),
    "f8_D50": (f8, np.full(50, -1.28), np.full(50,  1.28), 50),
    "f9_D10": (f9, np.full(10, -1.0),  np.full(10,  1.0), 10),
    "f9_D30": (f9, np.full(30, -1.0),  np.full(30,  1.0), 30),
    "f9_D50": (f9, np.full(50, -1.0),  np.full(50,  1.0), 50),
    "f10_D10": (f10, np.full(10, -100.0), np.full(10, 100.0), 10),
    "f10_D30": (f10, np.full(30, -100.0), np.full(30, 100.0), 30),
    "f10_D50": (f10, np.full(50, -100.0), np.full(50, 100.0), 50),
    "f11_D10": (f11, np.full(10, -1.28), np.full(10,  1.28), 10),
    "f11_D30": (f11, np.full(30, -1.28), np.full(30,  1.28), 30),
    "f11_D50": (f11, np.full(50, -1.28), np.full(50,  1.28), 50),
    "f12_D10": (f12, np.full(10, -1.28), np.full(10,  1.28), 10),
    "f12_D30": (f12, np.full(30, -1.28), np.full(30,  1.28), 30),
    "f12_D50": (f12, np.full(50, -1.28), np.full(50,  1.28), 50),
    "f13_D10": (f13, np.full(10, -5.12), np.full(10,  5.12), 10),
    "f13_D30": (f13, np.full(30, -5.12), np.full(30,  5.12), 30),
    "f13_D50": (f13, np.full(50, -5.12), np.full(50,  5.12), 50),
    "f14_D10": (f14, np.full(10, -32.0), np.full(10,  32.0), 10),
    "f14_D30": (f14, np.full(30, -32.0), np.full(30,  32.0), 30),
    "f14_D50": (f14, np.full(50, -32.0), np.full(50,  32.0), 50),
    "f15_D10": (f15, np.full(10, -600.0), np.full(10,  600.0), 10),
    "f15_D30": (f15, np.full(30, -600.0), np.full(30,  600.0), 30),
    "f15_D50": (f15, np.full(50, -600.0), np.full(50,  600.0), 50),
    "f16_D10": (f16, np.full(10, -100.0), np.full(10, 100.0), 10),
    "f16_D30": (f16, np.full(30, -100.0), np.full(30, 100.0), 30),
    "f16_D50": (f16, np.full(50, -100.0), np.full(50, 100.0), 50),
    "f17_D10": (f17, np.full(10, -5.0), np.full(10,  5.0), 10),
    "f17_D30": (f17, np.full(30, -5.0), np.full(30,  5.0), 30),
    "f17_D50": (f17, np.full(50, -5.0), np.full(50,  5.0), 50),
    "f18_D10": (f18, np.full(10, -10.0), np.full(10,  10.0), 10),
    "f18_D30": (f18, np.full(30, -10.0), np.full(30,  10.0), 30),
    "f18_D50": (f18, np.full(50, -10.0), np.full(50,  10.0), 50),
    "f19_D10": (f19, np.full(10, -5.12), np.full(10,  5.12), 10),
    "f19_D30": (f19, np.full(30, -5.12), np.full(30,  5.12), 30),
    "f19_D50": (f19, np.full(50, -5.12), np.full(50,  5.12), 50),
    "f20_D10": (f20, np.full(10, -50.0), np.full(10,  50.0), 10),
    "f20_D30": (f20, np.full(30, -50.0), np.full(30,  50.0), 30),
    "f20_D50": (f20, np.full(50, -50.0), np.full(50,  50.0), 50),
    "f21_D10": (f21, np.full(10, -500.0), np.full(10,  500.0), 10),
    "f21_D30": (f21, np.full(30, -500.0), np.full(30,  500.0), 30),
    "f21_D50": (f21, np.full(50, -500.0), np.full(50,  500.0), 50),
    "f22_D10": (f22, np.full(10, -5.12), np.full(10,  5.12), 10),
    "f22_D30": (f22, np.full(30, -5.12), np.full(30,  5.12), 30),
    "f22_D50": (f22, np.full(50, -5.12), np.full(50,  5.12), 50),
    "f23_D10": (f23, np.full(10, -50.0), np.full(10,  50.0), 10),
    "f23_D30": (f23, np.full(30, -50.0), np.full(30,  50.0), 30),
    "f23_D50": (f23, np.full(50, -50.0), np.full(50,  50.0), 50),
    "f24_D10": (f24, np.full(10, -10.0), np.full(10, 10.0), 10),
    "f24_D30": (f24, np.full(30, -10.0), np.full(30, 10.0), 30),
    "f24_D50": (f24, np.full(50, -10.0), np.full(50, 10.0), 50),
}
