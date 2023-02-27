# %%
import sys
sys.path.append("./expectation_value")
import numpy as np
import qiskit.quantum_info as qi
from utils_circuits import get_jksp_counts, state_r
from utils_strings import filtered_idxs

n_qubits = 5
di = 2**n_qubits
shots = 100000
r = 5
shots_r = 100
nbodies = [0, 2, 4]
phase = 0
# OJO CON LA SEED FIJA
state = qi.random_statevector(di, seed = 100).data
non_zero_idxs = state_r(n_qubits, state, r, shots=shots_r)
idxs = filtered_idxs(non_zero_idxs, nbodies)
strs, probs, ir = get_jksp_counts(n_qubits, state, idxs[:, 0], phase, shots=shots)
# %%
