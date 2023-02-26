# %%
import sys
sys.path.append("./expectation_value")
import numpy as np
import qiskit.quantum_info as qi
from expectationvalue import ExpVal


def random_pauli_nbodies(n_qubits, n_bodies, n_obs):
    """
    Generates random n_bodies Pauli strings.
    PARAMETERS:
        n_qubits: int. Number of qubits
        n_bodies: int. Total number of X + Y in the Pauli string. 
        n_obs: int. Total number of n_bodies Pauli strings. 
    RETURNS:
        observables: array (2, 2, n_qubits, n_obs). The first three indices are
            the Pauli string of n_qubits. 
    """
    X = np.array([[0., 1.], [1., 0.]], dtype="complex")
    Y = np.array([[0., -1.*1j], [1.*1j, 0.]], dtype="complex")
    Z = np.array([[1., 0], [0, -1.]], dtype="complex")
    I = np.array([[1., 0], [0, 1.]], dtype="complex")
    sigma = np.stack([I, X, Y, Z], axis=-1)
    obs = np.random.choice([0, 3], size=(n_qubits, n_obs)).astype(int)
    loc = np.array([np.random.choice(range(n_qubits), n_bodies, replace=False) 
                    for _ in range(n_obs)])
    for k in range(n_obs):
        for j in range(n_bodies):
            obs[loc[k, j], k] = np.random.choice([1, 2]).astype(int)
    observables = sigma[:, :, obs]
    return observables


n_qubits = 7
state = qi.random_statevector(2**n_qubits).data
# small_coeffs = np.random.choice(range(2**n_qubits), size=(2**n_qubits - 10,), 
#                                 replace=False)
# state[small_coeffs] = 10**(-4)*state[small_coeffs]
# state = state/np.linalg.norm(state)
r_shots = 1000
n_shots = 100000
r = 0.999
bodies = [0, 2, 4]
n_bodies = 2
n_obs = 500
obs = random_pauli_nbodies(n_qubits, n_bodies, n_obs)

ints = ExpVal(n_shots, bodies, r, r_shots, n_qubits)

ints.get_interferences(state)
ints.exp_val(obs)

ints.true_exp_val(obs)
print(np.abs(np.round(ints.e_val, 3) - np.round(ints.true_e_val, 3)))
print(np.sum(ints.e_val), np.sum(ints.true_e_val))
