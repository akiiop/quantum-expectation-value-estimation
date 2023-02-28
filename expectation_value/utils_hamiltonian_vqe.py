from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
import qiskit.quantum_info as qif

import numpy as np

import sys
sys.path.append("../expectation_value")
from expectationvalue import ExpVal



class Custom_VQE():
    """
    Custom VQE class that finds the ground state, evaluating the expectation value
    of the molecule hamiltonian using our method.
    
    Parameters
    ----------
    ansatz : QuantumCircuit.
         A parameterized quantum circuit to prepare the trial state.
    optimizer : Optimizer.
        A classical optimizer to find the minimum energy from
        the Qiskit :class:`.Optimizer`.
    initial_point : array.
        The initial point for the optimization. 
    total_shots : int.
        Total number of shots used in the expectation value measurement protocol.
    r_shots : int.
        Number of shots used to measure the computational basis 
        in the expectation value measurement protocol.
    r : int (0, 2**n_qubits) or float [0, 1].
        Number of coefficients that we 
        want to preserve or total probability of the preserved elements.
    """
    
    def __init__(self,
                 ansatz,
                 optimizer,
                 initial_point,
                 total_shots,
                 r_shots,
                 r
                ):
        
        self.initial_point = initial_point
        self.ansatz = ansatz
        self.optimizer = optimizer
        
        self.total_shots = total_shots
        self.r_shots = r_shots
        self.r = r
    
    def compute_minimum_eigenvalue(self, hamiltonian):
        """
        Compute the minimum eigenvalue of the hamiltonian.

        Parameters
        ----------
        hamiltonian : PauliSumOp.
            Hamiltonian.
        
        
        Returns
        -------
        exp_val_i : list.
            Returns the results obtained after every evaluation of the function exp_val_vqe, and will depend
            on the number of cost function evaluations of the optimizer method.
        state_i : list
            Returns the circuit used in every evaluation of the function exp_val_vqe, and will depend
            on the number of cost function evaluations of the optimizer method.
        final_result : OptimizerResult.
            The result of an optimization routine.
            
        """    
        
        # Save intermediate results
        exp_val_i = []
        state_i = []
        
        def exp_val_vqe(state, hamiltonian):
            
            """
            Compute the expectation value of the hamiltonian for a given state.

            Parameters
            ----------
            state : QuantumCircuit.
                The quantum circuit from the ansatz with the parameters
                following the optimization procedure.
            hamiltonian : PauliSumOp.
                Hamiltonian.


            Returns
            -------
            expectation_value : float.
                Expectation value.
            """
            
            bodies = get_number_nbody_terms(hamiltonian)
            n_qubits = hamiltonian.num_qubits
            
            coeffs, obs = get_obs_data(hamiltonian)
            
            algorithm = ExpVal(n_shots = self.total_shots - self.r_shots,
                    bodies = bodies, 
                    r = self.r, 
                    r_shots = self.r_shots,
                    n_qubits = n_qubits)
            
            algorithm.get_interferences(state)
            
            first_term_coeff = hamiltonian[0].coeffs[0].real
            
            exp_val_paulis = algorithm.exp_val(obs)
            expectation_value = np.sum(exp_val_paulis*coeffs).real + first_term_coeff
            
            exp_val_i.append(expectation_value)
            state_i.append(state)
            
            return expectation_value
        
        
        expectation_value = lambda x: exp_val_vqe(self.ansatz.bind_parameters(x), hamiltonian).real
        final_result = self.optimizer.minimize( expectation_value , self.initial_point )
        
        return exp_val_i, state_i, final_result


class MoleculeHamiltonian():
    """
    The MoleculeHamiltonian class uses qiskit_nature functions to easly obtain the qubit hamiltonian and
    groundstate/ground energy for a given molecule.
    
    """
    
    def __init__(self,
                 molecule,
                 converter = QubitConverter(JordanWignerMapper()),
                 basis = 'sto3g',
                ):
        """

        Parameters
        ----------
        molecule : string.
            Molecule geometry.
        converter : QubitConverter.
            Convert which transforms the second quantized hamiltonian to a qubit hamiltonian. By default,
            we use the JordanWignerMapper without reductions.
        basis : string.
            Basis set name as recognized by PySCF. We use sto3g as default, but others can be chosen
            from the list of PySCF’s valid basis set names 1_.
        
        . _1: https://pyscf.org/pyscf_api_docs/pyscf.gto.basis.html#module-pyscf.gto.basis
        """
        
        self.molecule = molecule
        self.basis = basis
        driver = PySCFDriver(
            atom= self.molecule,
            basis=self.basis,
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
            )                                  # Define the PySCFDriver for a given molecule
        
        self.driver = driver
        self.problem = driver.run()            # Define the ElectronicStructureProblem from the driver
        self.converter = converter
        
    def Hamiltonian(self,
                    freeze_core = False,
                    remove_orbitals = None):
        
        
        """
        Returns the qubit hamiltonian, allowing Active-Space reduction using the FreezeCoreTransformer to
        obtain a hamiltonian with a lower number of qubits.


        Parameters
        ----------
        freeze_core : bool.
            Choose to inactive and remove the core orbitals according to count_core_orbitals.
        remove_orbitals : List, optional.
            Choose to remove aditional orbitals.
            
        Returns
        -------
        qubit_op : PauliSumOp.
            Qubit hamiltonian.
        """
        problem = self.problem
        fermionic_op = self.problem.hamiltonian.second_q_op()
            
        if freeze_core or remove_orbitals is not None:
            fc_transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)
            problem = fc_transformer.transform(problem)
            fermionic_op = problem.hamiltonian.second_q_op()
            
        qubit_op = self.converter.convert(fermionic_op,
                                           sector_locator=problem.symmetry_sector_locator)
        return qubit_op
    
    def ComputeGroundState(self , circuit = True):
        
        """
        Compute the exact ground state and ground state energy using the NumPyMinimumEigensolver.

        Parameters
        ----------
        circuit : bool.
            If True, returns a quantum circuit that constructs the ground state. If False,
            returns the ground state as an array using Statevector.


        Returns
        -------
        state : QuantumCircuit or array(2**num_qubits)
            Ground state as a quantum circuit or as an array.
        eigenvalue : float.
            Ground state energy.
        """
        solver = NumPyMinimumEigensolver()
        
        ground_solver = GroundStateEigensolver(self.converter, solver)
        ground_state = ground_solver.solve(self.problem)
        
        state = ground_state.groundstate[0]
        eigenvalue = ground_state.groundenergy
        
        if circuit:
            return state, eigenvalue
        else:
            return qif.Statevector(state).data, eigenvalue
        
def get_number_nbody_terms(hamiltonian):
    """
    Gets the needed data of the hamiltonian to compute the expectation values using ExpVal.exp_val().


    Parameters
    ----------
    hamiltonian : PauliSumOp.
        Hamiltonian.


    Returns
    -------
    bodies : list.
        N-bodies interactions.
    """
    
    n_bodies_inter = []
    
    for i in range(len(hamiltonian)):
        pauli_string = hamiltonian[i].to_pauli_op().primitive.to_label()
        num_x = pauli_string.count('X')
        num_y = pauli_string.count('Y')
        
        n_body_terms = num_x+num_y
        
        n_bodies_inter.append(n_body_terms)
    
    bodies = list(set(n_bodies_inter))
    
    return bodies


def get_obs_data(hamiltonian):
    """
    Gets the needed data of the hamiltonian to compute the expectation values using ExpVal.exp_val().


    Parameters
    ----------
    hamiltonian : PauliSumOp.
        Hamiltonian.


    Returns
    -------
    coeffs : array(num_paulis).
        Coefficients of each pauli string of the hamiltonian.
    obs : array(2, 2, n_qubits, num_paulis)
        Observable made of every pauli string of the hamiltonian.
    """
    
    num_paulis = len(hamiltonian)
    n_qubits = hamiltonian.num_qubits

    paulis_dict = {'X': np.array([[0., 1.], [1., 0.]], dtype="complex"),
               'Y': np.array([[0., -1.*1j], [1.*1j, 0.]], dtype="complex"),
               'Z': np.array([[1., 0], [0, -1.]], dtype="complex"),
               'I': np.array([[1., 0], [0, 1.]], dtype="complex")}

    obs = np.zeros((2, 2, n_qubits, num_paulis), dtype=complex)
    coeffs = np.zeros(num_paulis, dtype=complex)
    
    # We remove the first pauli string which corresponds to the identity, 
    # as its trace with respect to any state is always 1.
    
    for i in range(1, num_paulis):
        pauli_string_coeff = hamiltonian[i].coeffs[0]
        pauli_string = hamiltonian[i].to_pauli_op().primitive.to_label()
        pauli_string_list = [paulis_dict[i] for i in pauli_string]
        obs[:,:,:,i] =  np.stack(pauli_string_list, axis=-1)
        coeffs[i] = pauli_string_coeff
    
    return coeffs, obs