"""Triangulation and observation selection, with no model downloads on import."""

import itertools
from pathlib import Path

import numpy as np


def select_views(images, excluded):
    return [im for im in images if Path(im["name"]).stem not in excluded]


def dlt(Ps, uvs):
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def reproj(P, X):
    x = P @ np.append(X, 1)
    return x[:2] / x[2]


def triangulate(Ps, uvs, thr_px=12.0):
    n = len(Ps)
    if n < 2:
        return None, np.inf, 0
    best = (None, np.inf, 0)
    for k in (2, 3):
        if n < k:
            break
        for sub in itertools.combinations(range(n), k):
            X = dlt([Ps[i] for i in sub], [uvs[i] for i in sub])
            err = np.array(
                [np.linalg.norm(reproj(Ps[i], X) - uvs[i]) for i in range(n)]
            )
            inl = err < thr_px
            if inl.sum() >= 2 and (
                inl.sum() > best[2]
                or (inl.sum() == best[2] and err[inl].mean() < best[1])
            ):
                Xr = dlt(
                    [Ps[i] for i in np.flatnonzero(inl)],
                    [uvs[i] for i in np.flatnonzero(inl)],
                )
                err2 = np.array(
                    [np.linalg.norm(reproj(Ps[i], Xr) - uvs[i]) for i in range(n)]
                )
                inl2 = err2 < thr_px
                best = (
                    Xr,
                    float(err2[inl2].mean()) if inl2.any() else np.inf,
                    int(inl2.sum()),
                )
    return best
