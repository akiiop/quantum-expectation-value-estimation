import numpy as np
from utils_strings import filtered_idxs
from utils_circuits import get_jksp_counts, state_r



class ExpVal():
    """
    Class that calculates the expectation value of many observables using data 
    collected from a quantum state. 

    Parameters
    ----------
    n_shots : int.
        Total number of shots used in the protocol.
    bodies: list
        Hamming distances of states |i> +- |j> that we want to measure. 
    r : int (0, 2**n_qubits) or float [0, 1].
        Number of coefficients that we 
        want to preserve or total probability of the preserved elements.
    shots_r : int.
        Number of shots used to measure the computational basis.
    n_qubits : int.
        Number of qubits of the system.
    """
    def __init__(self,
        n_shots,
        bodies, 
        r, 
        r_shots,
        n_qubits
        ):

        self.n_shots = n_shots
        self.bodies = bodies 
        self.r = r 
        self.r_shots = r_shots
        self.n_qubits = n_qubits
        self.interferences = None
        self.probs = None
        self.irs = None    

    def get_interferences(self, state):
        """
        Gets n_shots pairs of interferences for state, measuring in bases of the form
        |i> + exp(phase) |j>. It adds the following atributes to the ExpVal object. 
            
            a_interferences : array(2*n_qubits, unique_interferences_pairs).
                Gets an array of unique pairs of interferences. For each k in the first
                index, it contains two separable n_qubits state such that we can
                form <i|O|j> using them.
            a_probs : array(unique_interferences_pairs)
                Probabilities for each pair of interferences.
            a_irs : array(unique, n_shots, 2).
                For each unique pairs of interferences, gives the sign and phase of the 
                measured state |i> +  sign * exp(phase*pi/2) |j>. 


        Parameters
        ----------
        state : array(2**n_qubits).
            State.
        """
        self.state = state
        p, non_zero_idxs, R = state_r(self.n_qubits, self.state, self.r, 
                                      shots=self.r_shots)
        # generate every possible string
        lst_idxs = filtered_idxs(non_zero_idxs, self.bodies)       
        d = lst_idxs.shape[1]
        # select a random string
        idxs = lst_idxs[:, np.random.randint(0, d, self.n_shots)] 
        # add the phase at the end
        idxs = np.concatenate([idxs, np.random.choice([0, 1], 
                                                      size=(1, self.n_shots))])
        for k in range(self.n_shots):
            while int(np.sum(idxs[:self.n_qubits, k])==0) & (idxs[-1, k]==1):
                idxs[:, k] = np.concatenate([lst_idxs[:, np.random.randint(0, d)], 
                                            np.random.choice([0, 1], size=(1, ))])
        idxs, shots = np.unique(idxs, axis=1, return_counts=True)
        interferences = []
        irs = []
        probs = []
        for k in range(idxs.shape[1]):
            shadow, prob, ir = get_jksp_counts(self.n_qubits, state, idxs[:self.n_qubits, k], 
                                                idxs[-1, k], shots=shots[k])
            if int(np.sum(idxs[:self.n_qubits, k]))== 0:
                ir[:, 0] = 1
            interferences.append(shadow)
            irs.append(ir)
            probs.append(prob/shots[k])
        self.interferences = np.concatenate(interferences, axis=0, dtype=int)
        self.probs = np.concatenate(probs, axis=0, dtype=float)
        self.irs = np.concatenate(irs, axis=0, dtype=int)
            
    def exp_val(self, obs):
        """
        Takes an array obs of dimension (2, 2, n_qubits, n_obs). These are the 
        n_obs Pauli strings that compose the observable. Then, using
        self.interferences, self.probs and self.irs evaluates the expectation
        values. 

        Parameters
        ----------
        obs : array(2, 2, n_qubits, n_obs). 
            Observable that we want to measure, made 
            of n_qubits local single-qubit pauli matrices.
            
            
        Returns
        -------
        e_val : array (n_obs).
            The expectation values of all the Pauli strings in obs. 
        """
        n_basis = self.interferences.shape[0]
        n_obs = obs.shape[-1]
        idxi = self.interferences[:, :self.n_qubits]
        idxj = self.interferences[:, self.n_qubits:]
        R = np.zeros((n_basis, n_obs, self.n_qubits), dtype="complex")
        for k in range(self.n_qubits):
            R[:, :, k] = obs[idxj[:, k], idxi[:, k], k, :]
        pro = self.probs.reshape(-1, 1)*np.prod(R, axis=2)
        pro = ((1 - self.irs[:, 1].reshape(-1, 1))*np.real(pro) 
            + self.irs[:, 1].reshape(-1, 1)*np.imag(pro))
        ex = self.irs[:, 0].reshape(-1, 1)*pro
        e_val = np.sum(ex, axis=0)
        return e_val


    def true_exp_val(self, obs, state):
        """
        Calculates the true expectation value using the vectorized state. 
        
        Parameters
        ----------
        obs : array(2, 2, n_qubits, n_obs). 
            Observable that we want to measure, made 
            of n_qubits local single-qubit pauli matrices.
        state : array(2**n_qubits).
            State.
            
    
        Returns
        -------
        true_e_val : array(n_obs).
            The exact expectation values of all the Pauli strings in obs with the state. 
        """
        n_obs = obs.shape[-1]
        exact = np.zeros((n_obs))
        psi = state.reshape(-1, 1)
        for k in range(n_obs):
            obs_full = 1
            for j in range(self.n_qubits):
                obs_full = np.kron(obs_full, obs[:, :, j, k])
            exact[k] = np.real(psi.conj().T @ obs_full @psi)[0, 0]
        true_e_val = exact
        return true_e_val
