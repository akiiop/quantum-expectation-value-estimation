from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

import numpy as np



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