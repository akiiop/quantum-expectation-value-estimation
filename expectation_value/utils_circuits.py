from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from utils_strings import string2strings
import numpy as np


def basis_q(n_qubits, idx, phase):
    """
    Gets a the circuit of n_qubits that maps the state |0...0> to the 
    state |0...0> + exp(phase*pi/2) |idx>.
    
    
    Parameters
    ----------
    n_qubits : int.
        Number of qubits of the system.
    idx : array(n_qubits).
        Indices of the state |idx> to which |0...0> is mapped. 
    phase : float.
        Phase that is added to |0...0> + exp(phase*pi/2) |idx>. 
    
    
    Returns
    -------
        qc : QuantumCircuit.
            Quantum circuit that generates the state |0...0> + exp(phase*pi/2) |idx>.
        
    """
    qr = QuantumRegister(n_qubits, name="qr") # create a quantum register qr
    qc = QuantumCircuit(qr) # create quantum circuit with the previous registers
    idx = idx[::-1].astype(int)
    k = 0
    
    # Apply a hadamard gate in the first non-zero term of |idx>. 
    # If the phase is one, we prepare the state |0...0> + 1j |idx>. 
    for j in range(n_qubits):
        if idx[j]==1:
            qc.h(j)
            qc.p(phase*np.pi/2, j)
            k = j
            break
        else:
            pass
    
    # Chain of cnot gates. 
    for j in range(k+1, n_qubits):
        if idx[j]==1:
            qc.cnot(k, j)
            k = j
    return qc


# ACA SE PODRIA MODIFICAR EL get_data DE TAL FORMA QUE PODAMOS CAMBIAR EL BACKEND


def get_data(n_qubits, state, idx, phase, shots=1):
    """
    Given n_qubits, prepares a state and measures the base that contains 
    |0...0> + exp(phase*pi/2) |idx> a number of shots times.
    
    
    Parameters
    ----------
    n_qubits : int.
        Number of qubits of the system.
    state : array(2**n_qubits).
        State that we want to prepare.
    idx : array(n_qubits).
        Indices to which |0...0> is mapped. 
    phase : float.
        Phase that is added to |0..0> + exp(phase*pi/2) |idx>.
    shots : int.
        Number of shots that we want to measure.
        
    
    Returns
    -------
    counts : dict.
        Dictionary with the number of counts obtained
    """
    qr = QuantumRegister(n_qubits, name="qr")       # create a quantum register qr
    cr = ClassicalRegister(n_qubits, name="cr")     # create a classical register cr
    qc = QuantumCircuit(qr, cr)                     # create quantum circuit with the registers
    qc.initialize(state)                            # the state that we want to prepare
    base = basis_q(n_qubits, idx, phase)            # base contains |0..0> + e^(phase) |idx>
    qc.append(base.inverse(), qr)
    qc.measure(qr, cr)
    # now we measure
    backend = Aer.get_backend('aer_simulator')
    # backend = Aer.get_backend('aer_simulator_statevector')
    job = execute(qc, backend, shots=shots)
    result = job.result()
    counts = result.get_counts() #dictionary
    return counts


def get_jksp_counts(n_qubits, state, idx, phase, shots=1):
    """
    Given a quantum state, measures the basis defined by idx and phase, and 
    returns the strings of the states |i> and |j> in the basis,
    their associated counts, the sign and the phase. 
    
    
    Parameters
    ----------
    n_qubits : int.
        Number of qubits of the system.
    state : array(2**n_qubits).
        Quantum state
    idx : array(n_qubits).
        String to which zero is mapped
    phase : float.
        Phase that is added to |j> + exp(phase*pi/2)|k>.
    shots : int.
        Number of shots used to measure the basis
    
    
    Returns
    -------
    strs : array(n_strs, 2*n_qubits).
        Each row contains the strings of the sates |i> and |j>.
    probs : array(n_strs).
        Probabilities associated to the state |i> and |j>.
    ir : array (n_strs, 2).
        Sign and phase of |i> + |j>
    """
    counts = get_data(n_qubits, state, idx, phase, shots=shots)
    n_strs = len(counts)
    ir = np.zeros((n_strs, 2), dtype="int")
    data = np.zeros((n_strs), dtype="int")
    strs = np.zeros((n_strs, 2*n_qubits), dtype="int")
    k = 0
    for key, item in counts.items():
        data[k] = item
        s = np.array(list(key), dtype="int") # string that we measure
        st1, st2, sig, phase = string2strings(n_qubits, s, idx, phase)
        strs[k, 0:n_qubits] = st1
        strs[k, n_qubits:] = st2
        ir[k, 0] = sig
        ir[k, 1] = phase
        k += 1
    return strs, data, ir


def state_r(n_qubits, state, r, shots):
    """
    Gets the concentrated state, keeping the larger r values. 
    PARAMETERS:
        n_qubits: int. Number of qubits of the system.
        state: array (2**n_qubits, ). Quantum state
        r: int (0, 2**n_qubits) or float [0, 1]. Number of coefficients that we 
            want to preserve or total probability of the preserved elements. 
        shots: int. Number of shots used to measure the computational basis
    RETURNS:
        probs: array (2**n_qubits, ). Computational basis measurements probabilities
        non_zero_idxs: array (n_qubits, r). The non-zero kept elements of the concentrated state, ordered from larger to smaller 
        psi_r: array (2**n_qubits, ). Concentrated state
    """
    
    #preparation and measurement of the state
    qr = QuantumRegister(n_qubits, name="qr")
    cr = ClassicalRegister(n_qubits, name="cr")
    qc = QuantumCircuit(qr, cr)
    qc.initialize(state)
    qc.measure(qr, cr)
    backend = Aer.get_backend('aer_simulator')
    job = execute(qc, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    # ordering the counts from larger to smaller
    lst = [(k, v) for k, v in counts.items()]
    lst = sorted(lst, key=lambda x: x[1], reverse = True)
    # get arrays that contain the information
    idxs = np.zeros((n_qubits, len(lst)), dtype=int)
    counts = np.zeros((len(lst)), dtype=int)
    probs = np.zeros((2**n_qubits))
    for k in range(len(lst)):
        idxs[:, k] = np.array([int(x) for x in lst[k][0]])
        counts[k] = lst[k][1]
        probs[int(lst[k][0], 2)] = lst[k][1]
    probs = probs/np.sum(probs)
    psi_r = np.zeros((2**n_qubits))
    if r > 1:
        r = np.minimum(r, len(lst))
    else:
         r = np.sum(~(np.cumsum(counts) > shots*r))
    for k in range(r):
        psi_r[int(lst[k][0], 2)] = lst[k][1]
    psi_r = psi_r/np.linalg.norm(psi_r)
    non_zero_idxs = idxs[:, :r]

    return non_zero_idxs