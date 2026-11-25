# CEC 2027 Competition on Variational Quantum Optimization

Benchmark suite for the IEEE CEC 2027 competition on the classical optimization
loop of variational quantum algorithms (VQE and QAOA). Each problem is the exact
energy landscape of a fixed Hamiltonian–ansatz pair, evaluated by statevector
simulation: 15 functions in five groups, 25 function–size conditions, with
dimensions from 6 to 84.

The problem definitions and evaluation criteria are given in the technical
report (link to follow).

## Contents

| File | Purpose |
| --- | --- |
| `vqa_cec2027.py` | Reference evaluator (NumPy only) |
| `instances/F{k}_N{n}.json` | Frozen problem instances: Hamiltonian, ansatz, shift vector, periodic coordinates, reference energies |
| `Rand_Seeds.txt` | The 1000 run seeds |
| `example_run.py` | How to count FEs, record checkpoints and write the results file |

## Usage

```python
import numpy as np
import vqa_cec2027 as V

f = V.load("F7", 12)                 # function 7, N = 12 qubits
print(f.dim)                          # number of variables D
x = np.random.uniform(-np.pi, np.pi, f.dim)
print(f.error(x))                     # competition error e = (F - E_ref) / (E_max - E_0)

for c in V.catalogue():               # all conditions
    print(c)
```

Search range is `[-pi, pi]^D`. Periodic coordinates are wrapped and the others
are clipped by the evaluator. One call of `f.error` (or `f.energy`) is one
function evaluation.

```bash
python example_run.py F7 12          # writes results/RandomSearch_7_12.txt
```

## Rules in short

* Treat every problem as a black box: use only objective values, not the
  Hamiltonian, circuit, gradients or known symmetries.
* 30 runs per condition, seeds from `Rand_Seeds.txt`,
  MaxFES = max(10^4·D, 5·10^4); stop early if e ≤ 1e-8.
* One parameter setting for all functions.

Requirements: Python ≥ 3.10, NumPy.

## License

MIT; see `LICENSE`.
