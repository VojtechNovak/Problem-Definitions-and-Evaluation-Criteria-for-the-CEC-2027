"""Reference evaluator for the CEC 2027 VQA benchmark suite.

Every function is a frozen Hamiltonian--ansatz pair stored in
``instances/F{k}_N{n}.json``.  The evaluator is deliberately small, NumPy-only
and independent of the study code from which the instances were exported;
``tools/verify_against_originals.py`` checks it against that code.

Conventions
-----------
* Qubit q corresponds to bit ``1 << q`` of the computational-basis index.
* A Hamiltonian is a list of pure-axis Pauli strings (all factors X, all Y or
  all Z), which covers every model in the suite.
* R_A(t) = exp(-i t A / 2).  The HEA applies an R_y layer, then the CNOT chain
  0->1, 1->2, ..., N-2->N-1, then the next R_y layer; parameters are layer-major.
* QAOA coordinates are interleaved (gamma_1, beta_1, gamma_2, beta_2, ...).
  The physical mixer angle is beta = phi / 2, so phi has period 2*pi.
  Tied-reuse QAOA applies every layer ``tied_repeats`` times with equal angles.
* Block-QAOA coordinates per layer are (gamma_A, beta_A, gamma_B, beta_B);
  bridge terms are split equally between the two cost generators.
* Search box [-pi, pi]^D.  Coordinates flagged periodic are wrapped; the
  others (QAOA cost angles of Hamiltonians with non-integer spectra) are
  clipped to the box.
* The competition objective F(theta) = f(theta - o) uses the published shift
  vector o on periodic coordinates only (o is zero on the others).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
INSTANCE_DIR = HERE / "instances"


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------
def _z_signs(n: int) -> np.ndarray:
    idx = np.arange(1 << n, dtype=np.int64)
    return (1 - 2 * ((idx[:, None] >> np.arange(n)) & 1)).astype(np.float64)


class PauliHamiltonian:
    def __init__(self, n: int, terms: List[dict]):
        self.n = int(n)
        self.terms = terms
        self.axes = sorted({t["axis"] for t in terms})
        self._diag: Optional[Dict[str, np.ndarray]] = None
        self._component_z: Optional[Dict[str, np.ndarray]] = None

    def _build(self) -> None:
        if self.n > 22:
            raise MemoryError("dense diagonals are only built for N <= 22")
        z = _z_signs(self.n)
        self._diag, self._component_z = {}, {}
        for t in self.terms:
            axis = t["axis"]
            val = float(t["coeff"]) * np.prod(z[:, list(t["qubits"])], axis=1)
            self._diag[axis] = self._diag.get(axis, 0.0) + val
            if axis == "Z":
                comp = t.get("component", "full")
                self._component_z[comp] = self._component_z.get(comp, 0.0) + val
        popcount = np.array([bin(i).count("1") for i in range(1 << self.n)])
        self._y_phase = (-1j) ** popcount

    @property
    def diag(self) -> Dict[str, np.ndarray]:
        if self._diag is None:
            self._build()
        return self._diag

    @property
    def component_z(self) -> Dict[str, np.ndarray]:
        if self._component_z is None:
            self._build()
        return self._component_z

    @property
    def is_diagonal(self) -> bool:
        return set(self.axes) <= {"Z"}

    def _hadamard_all(self, v: np.ndarray) -> np.ndarray:
        v = v.astype(np.complex128).copy()
        for q in range(self.n):
            w = v.reshape(-1, 2, 1 << q)
            a, b = w[:, 0, :].copy(), w[:, 1, :].copy()
            w[:, 0, :] = (a + b) / np.sqrt(2.0)
            w[:, 1, :] = (a - b) / np.sqrt(2.0)
        return v

    def expectation(self, psi: np.ndarray) -> float:
        e = 0.0
        if "Z" in self.diag:
            e += float(np.dot(np.abs(psi) ** 2, self.diag["Z"]))
        if "X" in self.diag:
            e += float(np.dot(np.abs(self._hadamard_all(psi)) ** 2, self.diag["X"]))
        if "Y" in self.diag:
            self.diag  # ensure phases exist
            # Y = S H Z H S^dagger  ->  rotate by S^dagger then H.
            e += float(np.dot(np.abs(self._hadamard_all(psi * self._y_phase)) ** 2, self.diag["Y"]))
        return e

    def matvec(self, psi: np.ndarray) -> np.ndarray:
        """H |psi>, used only for spectra."""
        out = np.zeros_like(psi, dtype=np.complex128)
        idx = np.arange(1 << self.n, dtype=np.int64)
        z = _z_signs(self.n)
        for t in self.terms:
            mask = 0
            for q in t["qubits"]:
                mask |= 1 << int(q)
            c = float(t["coeff"])
            zprod = np.prod(z[:, list(t["qubits"])], axis=1)
            if t["axis"] == "Z":
                out += c * zprod * psi
            elif t["axis"] == "X":
                out += c * psi[idx ^ mask]
            else:  # Y string: Y|b> = i(-1)^b |1-b> per factor
                k = len(t["qubits"])
                src = idx ^ mask
                # amplitude moved from |src> to |idx>: prod_q i * z_q(src)
                out += c * (1j ** k) * np.prod(z[src][:, list(t["qubits"])], axis=1) * psi[src]
        return out


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------
def _ry(state: np.ndarray, angle: float, q: int) -> None:
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    w = state.reshape(-1, 2, 1 << q)
    a, b = w[:, 0, :].copy(), w[:, 1, :].copy()
    w[:, 0, :] = c * a - s * b
    w[:, 1, :] = s * a + c * b


def _cnot_chain_perm(n: int) -> np.ndarray:
    """Index map: new_state = state[perm] for the chain 0->1, ..., N-2->N-1."""
    idx = np.arange(1 << n, dtype=np.int64)
    src = idx.copy()
    # Apply gates in order; track for every output index its source index.
    for c in range(n - 2, -1, -1):
        t = c + 1
        flip = ((src >> c) & 1) == 1
        src = np.where(flip, src ^ (1 << t), src)
    return src


def _xrot_all(re_im: np.ndarray, beta: float, qubits) -> None:
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in qubits:
        w = re_im.reshape(-1, 2, 1 << q)
        a, b = w[:, 0, :].copy(), w[:, 1, :].copy()
        w[:, 0, :] = c * a + s * b
        w[:, 1, :] = s * a + c * b


class VQAFunction:
    """One frozen function--size condition."""

    def __init__(self, spec: dict, shifted: bool = True):
        self.spec = spec
        self.fid = spec["function"]
        self.n = int(spec["n_qubits"])
        self.ansatz = spec["ansatz"]
        self.type = self.ansatz["type"]
        self.dim = int(self.ansatz["dimension"])
        self.H = PauliHamiltonian(self.n, spec["terms"])
        self.periodic = np.asarray(spec["periodic"], dtype=bool)
        self.shift = np.asarray(spec.get("shift", np.zeros(self.dim)), dtype=float) if shifted else np.zeros(self.dim)
        ref = spec.get("reference", {})
        self.E0 = ref.get("E0")
        self.Emax = ref.get("Emax")
        self.Eref = ref.get("Eref")
        if self.type == "hea":
            self._perm = _cnot_chain_perm(self.n)
        if self.type in ("qaoa", "block_qaoa") and not self.H.is_diagonal:
            raise ValueError("QAOA functions require a diagonal cost Hamiltonian")
        if self.type == "block_qaoa":
            split = int(self.ansatz["split"])
            self._A, self._B = list(range(split)), list(range(split, self.n))
            cz = self.H.component_z
            zero = np.zeros(1 << self.n)
            bridge = cz.get("bridge", zero)
            self._phase_A = cz.get("A", zero) + 0.5 * bridge
            self._phase_B = cz.get("B", zero) + 0.5 * bridge

    # -- domain ------------------------------------------------------------
    def canonical(self, theta) -> np.ndarray:
        x = np.asarray(theta, dtype=float).copy()
        if x.shape != (self.dim,):
            raise ValueError(f"{self.fid}: expected {self.dim} variables, got {x.shape}")
        x = x - self.shift
        p = self.periodic
        x[p] = (x[p] + np.pi) % (2 * np.pi) - np.pi
        x[~p] = np.clip(x[~p], -np.pi, np.pi)
        return x

    # -- states --------------------------------------------------------------
    def state(self, x: np.ndarray) -> np.ndarray:
        n = self.n
        if self.type in ("product", "hea"):
            layers = 0 if self.type == "product" else int(self.ansatz["layers"])
            psi = np.zeros(1 << n)
            psi[0] = 1.0
            for layer in range(layers + 1):
                if layer:
                    psi = psi[self._perm]
                for q in range(n):
                    _ry(psi, x[layer * n + q], q)
            return psi
        psi = np.full(1 << n, 2.0 ** (-n / 2), dtype=np.complex128)
        if self.type == "qaoa":
            cost = self.H.diag["Z"]
            r = int(self.ansatz.get("tied_repeats", 1))
            for layer in range(int(self.ansatz["depth"])):
                g, b = x[2 * layer], 0.5 * x[2 * layer + 1]
                ph = np.exp(-1j * g * cost)
                for _ in range(r):
                    psi *= ph
                    _xrot_all(psi, b, range(n))
            return psi
        if self.type == "block_qaoa":
            for layer in range(int(self.ansatz["depth"])):
                gA, bA, gB, bB = x[4 * layer: 4 * layer + 4]
                psi *= np.exp(-1j * (gA * self._phase_A + gB * self._phase_B))
                _xrot_all(psi, 0.5 * bA, self._A)
                _xrot_all(psi, 0.5 * bB, self._B)
            return psi
        raise ValueError(self.type)

    def energy(self, theta, canonical_input: bool = False) -> float:
        x = np.asarray(theta, float) if canonical_input else self.canonical(theta)
        if self.type == "product" and self.H.is_diagonal:
            c = np.cos(x)
            e = 0.0
            for t in self.spec["terms"]:
                e += float(t["coeff"]) * float(np.prod(c[list(t["qubits"])]))
            return e
        return self.H.expectation(self.state(x))

    def error(self, theta) -> float:
        """Competition error value e = (F - E_ref) / (E_max - E_0)."""
        return (self.energy(theta) - self.Eref) / (self.Emax - self.E0)

    __call__ = error


def load(fid: str, n: int, shifted: bool = True, instance_dir: Optional[Path] = None) -> VQAFunction:
    d = Path(instance_dir) if instance_dir else INSTANCE_DIR
    spec = json.loads((d / f"{fid}_N{n}.json").read_text())
    return VQAFunction(spec, shifted=shifted)


def catalogue(instance_dir: Optional[Path] = None) -> List[dict]:
    d = Path(instance_dir) if instance_dir else INSTANCE_DIR
    out = []
    for p in sorted(d.glob("F*_N*.json")):
        s = json.loads(p.read_text())
        out.append({k: s[k] for k in ("function", "group", "name", "n_qubits")} | {"dimension": s["ansatz"]["dimension"]})
    return out
