# This file will call all the other files which will have to be in this folder
import numpy as np
from utils_strings import filtered_idxs, c_i_c_j, filter_ci_cj, extra_shadows, R_idxs
from utils_circuits import get_jksp_counts, state_r



class ExpVal():
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
        |i> + exp(phase) |j>.


        Parameters
        ----------
        state : array(2**n_qubits).
            State.
        n_qubits : int.
            Number of qubits of the system.
        n_shots : int.
            Number of interferences.
        r : int.
            Number of coefficients that we want to keep.
        shots_r : int.
            Number of shots used to measure the computational basis.


        Returns
        -------
        a_interferences : array(2*n_qubits, unique_interferences_pairs).
            Gets an array of unique pairs of interferences. For each k in the first
            index, it contains two separable n_qubits state such that we can
            form <i|O|j> using them.
        a_probs : array(unique_interferences_pairs)
            Probabilities for each pair of interferences.
        a_irs : array(unique, n_shots, 2).
            For each unique pairs of interferences, gives the sign and phase of the 
            measured state |i> +  sign * exp(phase*pi/2) |j>. 
        lst_idxs : array().
            List of possible strings after applying the filter according to psi_r and
            the bodies.
        """
        self.state = state
        p, non_zero_idxs, R = state_r(self.n_qubits, self.state, self.r, shots=self.r_shots)
        lst_idxs = filtered_idxs(non_zero_idxs, self.bodies) # genera todas las posibles strings
        #print(lst_idxs.shape)        
        d = lst_idxs.shape[1]
        idxs = lst_idxs[:, np.random.randint(0, d, self.n_shots)] # selecciona strings random
        #agregamos la fase al final
        idxs = np.concatenate([idxs, np.random.choice([0, 1], size=(1, self.n_shots))])
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
        # return interferences, probs, irs
            
    def exp_val(self, obs):
        """
        Takes the observable and the shadows and calculate the expected value.
        The last observable is assignated to qubit 0. (???)
        
        
        Parameters
        ----------
        obs : array(2, 2, n_qubits). 
            Observable that we want to measure, made 
            of n_qubits local single-qubit pauli matrices.
        shadows : array(unique_n_shadows, 2*n_qubits).
            Array of unique pairs of shadows. For each k in the first
            index, it contains two separable n_qubits state such that we can
            form <i|O|j> using them.
        probs : array(unique_shadows_pairs)
            Probabilities for each pair of shadows. 
        ir : array(unique_n_shadows, 2).
            Sign and phase of the |i> +  sign * exp(phase*pi/2) |j> of each
            unique pairs of shadows.


        Returns
        -------
        e_val : float.
            Expected value of the observable. 
        """
        n_qubits = self.interferences.shape[1]//2
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

    def get_shadows_R(self, state):
        """
        Gets n_shadows shadows for state, measuring in bases of the form
        |i> + exp(phase)|j>. 
        PARAMETERS:
            state: array (2**n_qubits). 
            n_qubits: int. Number of qubits of the system.
            n_shadows: int. Number of shadows that we want.
        RETURNS:
            shadows: array (n_shadows, 2, n_qubits). For each k in the first index,
                it contains two separable n_qubits state such that we can form
                <i|O|j> using them.
            ir: array (n_shadows, 2). It contains the signs that correspond to the 
                shadows and if we measured |i> + |j> or |i> + 1j|j>. 
        """
        self.state = state
        p, non_zero_idxs, R = state_r(self.n_qubits, self.state, self.r, self.r_shots)
        diag_shadows = np.concatenate([non_zero_idxs, non_zero_idxs], axis=0).T
        diag_probs = p/np.sum(p)
        diag_irs = np.zeros((diag_probs.shape[0], 2), dtype= "int")
        diag_irs[:, 0] = 1
        diag_irs[:, 1] = 0
        non_zero_idxs
        lst_idxs = R_idxs(non_zero_idxs, R) # genera todas las posibles strings
        d = lst_idxs.shape[1]
        idxs = lst_idxs[:, np.random.randint(0, d, self.n_shots)] # selecciona strings random
        #agregamos la fase al final
        idxs = np.concatenate([idxs, np.random.choice([0, 1], size=(1, self.n_shots))])
        for k in range(self.n_shots):
            while int(np.sum(idxs[:self.n_qubits, k])==0) & (idxs[-1, k]==1):
                idxs[:, k] = np.concatenate([lst_idxs[:, np.random.randint(0, d)], 
                                            np.random.choice([0, 1], size=(1, ))])
        idxs, shots = np.unique(idxs, axis=1, return_counts=True)
        shadows = []
        irs = []
        probs = []
        for k in range(idxs.shape[1]):
            shadow, prob, ir = get_jksp_counts(self.n_qubits, self.state, idxs[:self.n_qubits, k], 
                                                idxs[-1, k], shots=shots[k])
            if int(np.sum(idxs[:self.n_qubits, k]))== 0:
                ir[:, 0] = 1
            shadows.append(shadow)
            irs.append(ir)
            probs.append(prob/shots[k])
        a_shadows = np.concatenate(shadows, axis=0, dtype=int)
        a_probs = np.concatenate(probs, axis=0, dtype=float)
        a_irs = np.concatenate(irs, axis=0, dtype=int)
        new_shadows, new_counts, new_irs = extra_shadows(a_shadows, a_probs, a_irs, 
                                                        non_zero_idxs[:, 0], 
                                                        p[0]/np.sum(p))
        a_shadows = np.concatenate([a_shadows, new_shadows, diag_shadows], axis=0, dtype=int)
        
        a_probs = np.concatenate([a_probs, new_counts, diag_probs], axis=0, dtype=float)
        a_irs = np.concatenate([a_irs, new_irs, diag_irs], axis=0, dtype=int)
        self.interferences = a_shadows
        self.probs = a_probs
        self.irs = a_irs