# This file will call all the other files which will have to be in this folder
import numpy as np
from utils_strings import filtered_idxs
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
        self.obs = None
        self.e_val = None
        self.true_e_val = None
        self.state = None
    

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
        non_zero_idxs = state_r(self.n_qubits, self.state, self.r, shots=self.r_shots)
        lst_idxs = filtered_idxs(non_zero_idxs, self.bodies) # genera todas las posibles strings
        print(lst_idxs.shape)        
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
        eval : float.
            Expected value of the observable. 
        """
        self.obs = obs
        n_qubits = self.interferences.shape[1]//2
        n_basis = self.interferences.shape[0]
        n_obs = self.obs.shape[-1]
        idxi = self.interferences[:, :self.n_qubits]
        idxj = self.interferences[:, self.n_qubits:]
        R = np.zeros((n_basis, n_obs, self.n_qubits), dtype="complex")
        for k in range(self.n_qubits):
            R[:, :, k] = self.obs[idxj[:, k], idxi[:, k], k, :]
        pro = self.probs.reshape(-1, 1)*np.prod(R, axis=2)
        pro = ((1 - self.irs[:, 1].reshape(-1, 1))*np.real(pro) 
            + self.irs[:, 1].reshape(-1, 1)*np.imag(pro))
        ex = self.irs[:, 0].reshape(-1, 1)*pro
        self.e_val = np.sum(ex, axis=0)

    def true_exp_val(self, obs):
        self.obs = obs
        n_obs = self.obs.shape[-1]
        exact = np.zeros((n_obs))
        psi = self.state.reshape(-1, 1)
        for k in range(n_obs):
            obs_full = 1
            for j in range(self.n_qubits):
                obs_full = np.kron(obs_full, self.obs[:, :, j, k])
            exact[k] = np.real(psi.conj().T @ obs_full @psi)[0, 0]
        self.true_e_val = exact