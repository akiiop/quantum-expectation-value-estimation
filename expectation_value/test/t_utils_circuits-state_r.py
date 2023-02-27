# %%
import sys
sys.path.append("./expectation_value")
import numpy as np
import qiskit.quantum_info as qi
from utils_circuits import state_r

n_qubits = 4
di = 2**n_qubits
r = 5
shots_r = 100
state = qi.random_statevector(di).data
non_zero_idxs = state_r(n_qubits, state, r, shots=shots_r)
print(non_zero_idxs)