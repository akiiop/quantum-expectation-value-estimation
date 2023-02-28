import sys

sys.path.append("../")

from expval_basissampling import BasisSamplingExpectation
import numpy as np
import qiskit.quantum_info as qi

from qiskit import Aer

from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

class MoleculeHamiltonian():
    def __init__(self,
                 molecule,
                 converter,
                ):
        self.molecule = molecule
        driver = PySCFDriver(
            atom= self.molecule,
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
            )
        
        self.es_problem = driver.run()
        self.converter = converter
        
    def Hamiltonian(self,
                    freeze_core = True,
                   ):
        es_problem = self.es_problem
        fermionic_op = self.es_problem.hamiltonian.second_q_op()
            
        if freeze_core:
            fc_transformer = FreezeCoreTransformer(freeze_core=True, remove_orbitals=None)
            es_problem = fc_transformer.transform(es_problem)
            fermionic_op = es_problem.hamiltonian.second_q_op()
            
        fc_qubit_op = self.converter.convert(fermionic_op,
                                           sector_locator=es_problem.symmetry_sector_locator,
                                          )
        return fc_qubit_op
    
    def ComputeGroundState(self , circuit = True):
        solver = NumPyMinimumEigensolver()
        
        ground_solver = GroundStateEigensolver(self.converter, solver)
        ground_state = ground_solver.solve(self.es_problem)
        
        state = ground_state.groundstate[0]
        eigenvalue = ground_state.groundenergy
        if circuit:
            return state , eigenvalue
        else:
            return qi.Statevector(state).data , eigenvalue

        
dist = 1.0
molecules = {
    "H2Be": "Be .0 .0 .0; H .0 .0 -" + str(dist) + "; H .0 .0 " + str(dist),
    "H2":"H .0 .0 .0; H .0 .0 " + str(dist),
    "LiH":"Li .0 .0 .0; H .0 .0 " + str(dist),
}




#Hamiltonian Construction
from tqdm import tqdm
converter = QubitConverter(JordanWignerMapper())
molecular_hamiltonian = MoleculeHamiltonian(molecules["LiH"] , converter)
hamiltonian = molecular_hamiltonian.Hamiltonian(freeze_core=False)
#State Construction
state, eig = molecular_hamiltonian.ComputeGroundState(circuit=False)
print("qubits = ", hamiltonian.num_qubits)
#basis sampling setting

r = 300
shots_total = 10**7
quantum_instance = QuantumInstance( Aer.get_backend( "qasm_simulator" ) )

expval = BasisSamplingExpectation( hamiltonian , quantum_instance )

reps = 400

circ = QuantumCircuit( hamiltonian.num_qubits )
circ.initialize(state)
data = expval.GetExpectation( circ,
                              r_shots = 1000,
                              total_shots=10**6,
                              allocate_shots=True,
                              R= 200 ).real

print("japos=",japos,"teor=",eig )