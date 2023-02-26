import numpy as np
import itertools

def cxor(s, k, j):
    """
    Classical cnot. Takes a string s, and depending on the control bit k, changes 
    the parity of the target bit j in s.
    
    
    Parameters
    ----------
    s: array. 
        Strig.
    k: int.
        Control bit.
    j: int.
        Target bit.
        
        
    Returns
    -------
    scxor : array.
        If bit k is zero, it scxor is the string as s. 
    """
    scxor = s.copy().astype(int)
    if s[k]==1:
        scxor[j] = (scxor[j] + 1)%2
    else:
        pass
    return scxor


# Estos no cacho muy bien que hacen

def sup_strings(st, idx, n_qubits):
    """
    From a string st, gives you the two strings that you get from the circuit 
    idx. st[0] is the most significant digit.
    
    
    Parameters
    ----------
    st : array(n_qubits).
        Original string.
    idx : array(n_qubits).
        Defines the circuit.
    n_qubits : int.
        Number of qubits of the system.
        
        
    Returns
    -------
    st1 : array(n_qubits).
        st string modified by the cnots.
    st2 : array(n_qubits).
        st with X applied modified by the cnots.
    k1 : int.
        First index with a 1 in idx. 
    """
    st1 = st.copy().astype(int)
    st2 = st.copy().astype(int)
    idx = idx.astype(int)
    k = 0
    for j in range(n_qubits):
        if idx[j]==1:
            st2[j] = (st2[j] + 1)%2
            k = j
            k1 = j
            break
        else:
            k1 = int(n_qubits-1)
    for j in range(k + 1, n_qubits):
        if idx[j]==1:
            st1 = cxor(st1, k, j)
            st2 = cxor(st2, k, j)
            k = j
        else:
            pass
    return st1, st2, k1


# Estos no cacho muy bien que hacen


def sign_string(s1, s2, st1, phase, k1):
    """
    Gets the sign that the relative between st1 and st2 should have, based on
    the term to which the circuit applied the hadamard gate, and also the 
    phase.
    
    
    Parameters
    ----------
    s1 : str.
        Binary representation of st1.
    s2 : str.
        Binary representation of st2.
    st1 : array(n_qubits).
        First element of the superposition. 
    phase : int.
        Encodes the phase of the basis.
    k1 : int.
        Bit to which we apply hadamard gate in the circuit. 
    
    
    Returns
    -------
    sign : int.
        The relative sign, st1 + sign*st2. 
    """
    sign = 1
    if phase==1:
        if s1 < s2:
            return sign
        else:
            return -sign    
    else:
        if st1[k1]==0:
            return sign   
        else:
            return -sign

# Estos no cacho muy bien que hacen, falta


def string2strings(n_qubits, st, idx, phase):
    """
    For a string, gives its classical |i> +- (real or imag)|j> superposition 
    using idx.
    
    
    Parameters
    ----------
    n_qubits : int.
        Number of qubits of the system.
    idx : list (n_qubits).
        Indices to which 0...0 is mapped. 
    st : string.
        
        
    Returns
    -------
    st_1, st_2 : strings.
        qc converts st into st1, st2 such that in the
        circuit |st> --> |st1> +- |st2>. 
    sign: int -1 or 1. 
    phase: int 0 or 1. 
    """
    phase = int(phase)
    idx = idx[::-1].astype(int)
    st = st[::-1].astype(int)
    st1, st2, k1 = sup_strings(st, idx, n_qubits) #little endian form
    s1 = '0b'+''.join(map(str, st1[::-1]))
    s2 = '0b'+''.join(map(str, st2[::-1]))
    if s2==s1:
        sign = 1
        phase = 0
        return st1[::-1], st2[::-1], sign, phase
    elif s1 < s2:
        sign = sign_string(s1, s2, st1, phase, k1)
        return st1[::-1], st2[::-1], sign, phase
    else:
        sign = sign_string(s1, s2, st1, phase, k1)
        return st2[::-1], st1[::-1], sign, phase

##############################################################################################

# falta mejorar de aqui pa abajo las explicaciones


def filter_hd(idx, hd):
    """
    Calculates the Hamming distance of 0..0 with idx and compares it with hd.
    
    
    Parameters
    ----------
    idx : array(n_qubits).
        Indices.
    hd : int.
        Desired Hamming distance
    
        
    Returns
    -------
    eq_hd : bool.
        True if the Hamming distance of idx is equal to hd.
    """
    hd_idx = int(np.sum(idx))
    eq_hd = hd_idx == int(hd)
    return eq_hd, idx


def pair_strings2zero(pair_strings):
    
    """
    Transforms two strings to the form 0...0 + idx_transf.
    
    
    Parameters
    ----------
    pair_strings : array(n_qubits, 2).
        Strings which represents the state |j> + |k>. 
        
        
    Returns
    -------
    idx_transf : array(n_qubits).
        The transformed string idx_transf.
    """
    
    n_qubits = pair_strings.shape[0]
    pair_strings = pair_strings.astype(int)
    idx_transf = np.copy(pair_strings[:, 1])
    for j in range(n_qubits):
        if pair_strings[j, 0] == 1:
            idx_transf[j] = (idx_transf[j] + 1) % 2
    return idx_transf


def filtered_idxs(non_zero_idxs, nbodies):
    """
    Given a list of non_zero idxs, gives the list of idxs from which we have to sample
    
    
    Parameters
    ----------
    non_zero_idxs : array(n_qubits, r).
        The non-zero kept elements of the concentrated state,
        ordered from larger to smaller.
    nbodies : list.
        Number of bodies interactions.
    
    
    Returns
    -------
    filtered_idxs : array(n_qubits, ?).
        Filtered strings according to the number of bodies interactions.
    """

    n_qubits, r = non_zero_idxs.shape
    combinations = np.zeros((n_qubits, 2, r**2))
    idxs = np.zeros((n_qubits, r**2))
    for j in range(r):
        for k in range(r):
            combinations[:, 0, r*j + k] = non_zero_idxs[:, j]
            combinations[:, 1, r*j + k] = non_zero_idxs[:, k]
    for j in range(r**2):
        idxs[:, j] = pair_strings2zero(combinations[:, :, j]) 
    idxs = np.unique(idxs, axis = 1)
    lst = []
    for k in range(idxs.shape[1]):
        for j in nbodies:
            eq_hd, idx = filter_hd(idxs[:, k], j)
            if eq_hd == True:
                lst.append(idx)         
    filtered_idxs = np.array(lst).T
    return filtered_idxs