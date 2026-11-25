"""Minimal example: run an optimizer on one condition and write the results file.

Usage:  python example_run.py F7 12
Writes  results/RandomSearch_7_12.txt  -- a 15 x 30 matrix:
rows 1-14 = error at the CEC 2017 checkpoints, row 15 = FE_term; columns = runs.
Replace `random_search` with your algorithm. Only objective values may be used.
"""
import sys
from pathlib import Path

import numpy as np

import vqa_cec2027 as V

CHECKPOINTS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RUNS = 30
TARGET = 1e-8



class Recorder:
    """Counts FEs and records best-so-far error at the checkpoints."""

    def __init__(self, f, max_fes):
        self.f, self.max_fes = f, max_fes
        self.fes, self.best, self.fe_term = 0, np.inf, max_fes
        self.marks = [int(np.floor(k * max_fes)) for k in CHECKPOINTS]
        self.record = []

    def __call__(self, x):
        if self.fes >= self.max_fes:
            raise StopIteration
        e = self.f.error(x)
        self.fes += 1
        if e < self.best:
            self.best = e
            if e <= TARGET and self.fe_term == self.max_fes:
                self.fe_term = self.fes
        while len(self.record) < len(self.marks) and self.fes >= self.marks[len(self.record)]:
            self.record.append(self.best)
        if self.best <= TARGET:
            raise StopIteration
        return e

    def finish(self):
        while len(self.record) < len(self.marks):
            self.record.append(max(self.best, TARGET) if self.best > TARGET else TARGET)
        return [max(v, TARGET) if v <= TARGET else v for v in self.record] + [self.fe_term]


def random_search(obj, dim, rng):
    while True:
        obj(rng.uniform(-np.pi, np.pi, dim))


def main():
    fid, n = sys.argv[1], int(sys.argv[2])
    f = V.load(fid, n)
    max_fes = max(10_000 * f.dim, 50_000)
    seeds = [int(s) for s in Path("Rand_Seeds.txt").read_text().split()]
    sizes = sorted({c["n_qubits"] for c in V.catalogue() if c["function"] == fid})
    s = sizes.index(n) + 1
    i = int(fid[1:])
    cols = []
    for r in range(1, RUNS + 1):
        seed = seeds[((i - 1) * 3 * 30 + (s - 1) * 30 + r - 1) % 1000]
        rec = Recorder(f, max_fes)
        try:
            random_search(rec, f.dim, np.random.default_rng(seed))
        except StopIteration:
            pass
        cols.append(rec.finish())
        print(f"run {r:2d}: final error {cols[-1][-2]:.3e}")
    Path("results").mkdir(exist_ok=True)
    np.savetxt(f"results/RandomSearch_{i}_{n}.txt", np.array(cols).T, fmt="%.10e")


if __name__ == "__main__":
    main()
