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



def filter_ci_cj(shadows, counts, irs, idx_c_i):
    n_shadows = shadows.shape[0]
    n_qubits = shadows.shape[1]//2
    new_shadows = []
    new_counts = []
    new_irs = []
    pos_ci = []
    for k in range(n_shadows):
        a = shadows[k, :n_qubits]
        b = shadows[k, n_qubits:]
        if np.array_equal(a, idx_c_i):
            new_shadows.append(b)
            new_counts.append(counts[k])
            new_irs.append(irs[k, :])
            pos_ci.append(1)
        else:
            pass
        if np.array_equal(b, idx_c_i):
            new_shadows.append(a)
            new_counts.append(counts[k])
            new_irs.append(irs[k, :])
            pos_ci.append(-1)
        else:
            pass
    new_shadows = np.array(new_shadows, dtype="int")
    new_counts = np.array(new_counts)
    new_irs = np.array(new_irs, dtype="int")
    return new_shadows, new_counts, new_irs, pos_ci


def c_i_c_j(shadows, counts, irs, idx_c_i):
    new_shadows, new_counts, new_irs, pos_ci = filter_ci_cj(shadows, counts, irs, idx_c_i)
    n_shadows = new_shadows.shape[0]
    u_shadows = np.unique(new_shadows, axis=0)
    n_u_shadows, n_qubits = u_shadows.shape
    cicj = np.zeros((n_u_shadows), dtype="complex")
    for j in range(n_u_shadows):
        for k in range(n_shadows):
            if np.array_equal(u_shadows[j, :], new_shadows[k, :]):
                A = (1 - new_irs[k, 1])*new_counts[k] - 1j*pos_ci[k]*new_irs[k, 1]*new_counts[k]
                cicj[j] += new_irs[k, 0]*A/2
            else:
                pass
    return  cicj, u_shadows


def extra_shadows(shadows, counts, irs, idx_ci, p_c_i):
    cicj, u_shadows = c_i_c_j(shadows, counts, irs, idx_ci)
    sidx = '0b'+''.join(map(str, idx_ci))
    n_cicj, n_qubits = u_shadows.shape
    n_ckcj = n_cicj**2
    ckcj = np.zeros(((n_ckcj-n_cicj)//2,), dtype="complex")
    shadows = np.zeros(((n_ckcj-n_cicj)//2, 2*n_qubits), dtype="int")
    i = 0
    for j in range(n_cicj):
        for k in range(j+1, n_cicj):
            su_shadowj = '0b'+''.join(map(str, u_shadows[j, :]))
            su_shadowk = '0b'+''.join(map(str, u_shadows[k, :]))
            if su_shadowj < su_shadowk:
                shadows[i, :n_qubits] = u_shadows[j, :]
                shadows[i, n_qubits:] = u_shadows[k, :]
                ckcj[i] = cicj[j]*np.conj(cicj[k])/p_c_i
            else:
                shadows[i, :n_qubits] = u_shadows[k, :]
                shadows[i, n_qubits:] = u_shadows[j, :]
                ckcj[i] = cicj[k]*np.conj(cicj[j])/p_c_i
            i += 1 
    l = shadows.shape[0]
    new_shadows = np.zeros((2*l, 2*n_qubits), dtype="int")
    new_irs = np.zeros((2*l, 2), dtype="int")
    new_counts = np.zeros((2*l, ), dtype="float")
    new_shadows[:l, :] = shadows
    new_shadows[l:, :] = shadows
    new_counts[:l] = 2*np.real(ckcj)
    new_counts[l:] = 2*np.imag(ckcj)
    new_irs[:l, 1] = 0
    new_irs[l:, 1] = 1
    new_irs[:l, 0] = 1
    new_irs[l:, 0] = 1
    return new_shadows, new_counts, new_irs

def R_idxs(non_zero_idxs, R):
    """
    Given a list of non_zero idxs, gives the list of idxs from which we have to sample
    """

    n_qubits, r = non_zero_idxs.shape[0], R
    combinations = np.zeros((n_qubits, 2, r - 1))
    idxs = np.zeros((n_qubits, r-1))
    for j in range(r-1):
            combinations[:, 0, j] = non_zero_idxs[:, 0]
            combinations[:, 1, j] = non_zero_idxs[:, j+1]
    for j in range(r-1):
        idxs[:, j] = pair_strings2zero(combinations[:, :, j]) 
    idxs = np.unique(idxs, axis = 1)
    return idxs


def diff_bit( bit1 , bit2 ):
    return  [ idx for idx , ( b1 , b2 ) in enumerate (zip( bit1,bit2 ) ) if b1!=b2   ]


