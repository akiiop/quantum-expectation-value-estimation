from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance
import qiskit.opflow as of
import numpy as np
from utils_strings import diff_bit

class BasisSamplingExpectation():
    def __init__( self,
                 operator,
                 quantum_instance,
                ):
        """
        Estimates the expectation value of an observable OPERATOR on the state given by the circuit
        CIRCUIT_STATE. The samplings are performed on the QUANTUM_INSTANCE. The tolerance indicates
        the minimum proability allowed for a basis to be included on the R elements
        operator: OpFlow.OperatorBasis
        quantum_instance: QuantumInstace or Backend
        """
        self.quantum_instance = (
            quantum_instance if isinstance(quantum_instance,QuantumInstance) else
            QuantumInstance(backend = quantum_instance) 
        )
        self.operator = operator
        
        
        
    def GetExpectation(self, 
                       circuit_state,
                       r_shots,
                       total_shots,
                       w=0.4,
                       allocate_shots=True,
                       R = 5):
        
        circuit_state.add_register( ClassicalRegister( circuit_state.num_qubits ) )
        
        if allocate_shots:
            f_shots, A_shots, B_shots, transition_matrix = self.ComputeResources( circuit_state,
                                                                            r_shots,
                                                                            total_shots,
                                                                            w,
                                                                            R,
                                                                           )
            weight_factors,R = self.WeightFactors( circuit_state, f_shots, R )
            weight_factors,R, A_shots , B_shots = self.check_factors_size( weight_factors,
                                                                          A_shots,
                                                                          B_shots,
                                                                         )
        else:
            f_shots = total_shots/R
            A_shots = total_shots/(2*R)
            B_shots = total_shots/(2*R)
            weight_factors,R = self.WeightFactors( circuit_state, f_shots, R )
            transition_matrix= self.TransitionMatrix( weight_factors.keys() )
                        
        #interference factors, a dictionary with the same keys of weight_factors
        # already the off diagonal terms are already divided by sqr(f_l)
        interference_factors = self.InterferenceFactors( weight_factors,
                                                        circuit_state,
                                                        A_shots,
                                                        B_shots,
                                                       ) 
        
        #computes the new normalization
        norm_square = 1 / np.sum(
            list(weight_factors.values())
        )
        
        # Creates a matrix from the weights_factors, is the poduct of all its elements
        weight_matrix = np.array( [ [ 
            val_i*val_j for val_j in weight_factors.values()   
        ]  for val_i in weight_factors.values() ] )
        
        # Creates a matrix from the interference_factors, similar manner than from the weights 
        # except for the diagonal
        n = len(interference_factors)
        interference_matrix = np.zeros( (n,n) , dtype = complex )
        for (i,val_i) in enumerate(weight_factors):
            
            for (j,val_j) in enumerate(weight_factors):
                
                if i!=j:
                    interference_matrix[i,j] = (
                        interference_factors[val_j]*interference_factors[val_i].conjugate()
                    )
                    
                else:
                    interference_matrix[ i,j ] = weight_factors[val_i]
        exp_val = norm_square*weight_matrix * transition_matrix / interference_matrix
        
        return exp_val.sum()
        
        
    def WeightFactors(self, circuit_state,shots,R):
        nq = circuit_state.num_qubits
        temp_circ = QuantumCircuit( nq , nq )
        
        temp_circ.compose( circuit_state , inplace=True )
        temp_circ.measure( temp_circ.qubits, temp_circ.clbits )
        
        trans_circ = self.quantum_instance.transpile( temp_circ )[0]
        self.quantum_instance.run_config.shots = shots
        results = self.quantum_instance.execute( trans_circ )
        results = results.get_counts()
        
        results = {  k:v/self.quantum_instance.run_config.shots for k,v in results.items() }
        if R>len(results):
            R = len(results)
            return dict(sorted( results.items() , key = lambda x:x[1], reverse=True )[:R]),R
        else:
            return dict(sorted( results.items() , key = lambda x:x[1], reverse=True )[:R]),R
            
    
    def TransitionMatrix( self , basis ):
        states = [ of.StateFn(el) for el in basis ]
        return np.array([[ (i.adjoint()@self.operator@k).eval()   for i in states] for k in states] )
        
    def InterferenceFactors( self, weights , circuit_state, A_shots, B_shots ):
        
        states =list(weights.keys())
        state_l = states[0]
        interference_factors = { state_l : weights[state_l]  }
        for i,state in  enumerate(states[1:]):
            if A_shots[i]==0:
                A=0
            else:
                A = self.InterferenceCircuitEvaluation( state_l,
                                                       state,
                                                       circuit_state,
                                                       A_shots[i],
                                                       phase = True,
                                                      )
                
            if B_shots[i]==0:
                B=0
            else:
                B = self.InterferenceCircuitEvaluation( state_l,
                                                       state,
                                                       circuit_state,
                                                       B_shots[i],
                                                       phase = False,
                                                      )
    
            interference_factors[state] =  A + B - (.5+.5j)*( 
                weights[state_l] + weights[state]
            )/np.sqrt( weights[state_l] )
        
        return interference_factors
    
    def InterferenceCircuitEvaluation( self, 
                                      m,
                                      n,
                                      circuit_state,
                                      shots,
                                      phase,
                                     ):
        """
        m , n:: binary representation 
        phase :: boolean
        Return circuit  for estimating A and B
        """
        

        temp_circuit = QuantumCircuit( len(m) , len(m) )
        temp_circuit.compose(circuit_state , inplace=True)
        
        qbits = diff_bit( m , n )
        
        m = m[::-1]
        for qbit , value in enumerate(m):
            if value =='1':
                temp_circuit.x(qbit)

        for qbit in qbits[:-1]:
            temp_circuit.cnot(qbit,qbit+1)

        if phase:
            temp_circuit.s(qbits[0])

        temp_circuit.h(qbits[0])

        if phase:
            temp_circuit.x(qbits[0])
        
        temp_circuit.measure( temp_circuit.qubits , temp_circuit.clbits )
        trans_circ = self.quantum_instance.transpile( temp_circuit )
        self.quantum_instance.run_config.shots = shots
        
        results = self.quantum_instance.execute( trans_circ ).get_counts()
        results.setdefault( '0'*len(m), 0.)
        
        return results['0'*len(m)]/self.quantum_instance.run_config.shots 
    
    def ComputeResources(self,
                         circuit_state,
                         r_shots,
                         total_shots,
                         w,
                         R,
                        ):

        basis , R= self.WeightFactors(circuit_state, r_shots,R)
        O = self.TransitionMatrix(basis.keys())
        f = np.array( [w]+(R-1)*[(1-w)*(R-1)] )
        A = 0.5*np.ones( R-1 )*(np.abs( np.sqrt(w) + np.sqrt((1-w)/(R-1))))**2
        B = 0.5*np.ones( R-1 )*(np.abs( np.sqrt(w) + 1j*np.sqrt((1-w)/(R-1))))**2
        
        
        partial_O_f = np.array(
            [ O[0,0] + ( (3+1j) * O[0,0] ).real + 
             + ( (3+1j)*np.sqrt( (1-w)/(w*(R-1)) )* O[:,1].sum() ).real
             + (2/w)*((1-w)/(R-1))*( 1+np.sqrt( (w*(R-1))/(1-w) ) )*(
                 sum( [sum([O[i,j] for j in range(i,R)]) for i in range(1,R)] ) 
                                                                    ).real
            ]
            + [
                O[r,r] + 
                ((2*np.sqrt(w*((R-1)/(1-w)))+1+1j)+O[1,r]).real+ 
                (
                    (2+((1-1j)*np.sqrt((1-w)/(w*(R-1)))))*
                    (O[r,:].sum()-O[r,r])
                ).real for r in range(1,R)
            ]
        )
        
        partial_O_A = -2*np.array( [
            O[0,r] + ( np.sqrt( ( 1-w )/( w*(R-1) ) * ( O[:,r].sum()-O[r,r] ) ) )
            for r in range(1,R)
        ]).real
        
        
        partial_O_B = 2*np.array( [
            O[0,r] + ( np.sqrt( ( 1-w )/( w*(R-1) ) * ( O[:,r].sum()-O[r,r] ) ) )
            for r in range(1,R)
        ]).imag
        
        
        v_f = ((partial_O_f**2*f*(1-f)).sum()).real
        v_A = (partial_O_A**2*A*(1-A)).real
        v_B = (partial_O_B**2*B*(1-B)).real
        
        v_sqrt = np.sqrt(v_f)+ np.sqrt(v_A).sum()+np.sqrt(v_B).sum()
        
        f_shots = np.ceil(total_shots*np.sqrt(v_f)/v_sqrt)
        A_shots = np.ceil(total_shots*np.sqrt(v_A)/v_sqrt)
        B_shots = np.ceil(total_shots*np.sqrt(v_B)/v_sqrt)
        

        return f_shots, A_shots, B_shots, O
    
    def check_factors_size(self, weight_factors, A_shots, B_shots):
        a_size = len(A_shots) + 1
        if len(weight_factors)==len(A_shots)+1:
            return weight_factors,a_size,A_shots,B_shots
        elif len(weight_factors)>a_size:
            weight_factors = dict( 
                sorted( weight_factors.items() , key = lambda x:x[1], reverse=True )[:a_size] 
            )
            return weight_factors, a_size,A_shots, B_shots
        elif len(weight_factors)<a_size:
            R = len(weight_factors)
            a_total_shots = A_shots.sum()
            b_total_shots = B_shots.sum()
            
            a_reduced_shots = A_shots[:R].sum()
            b_reduced_shots = B_shots[:R].sum()
            
            A_shots = A_shots[:R]*np.ceil( a_reduced_shots/a_total_shots )
            B_shots = B_shots[:R]*np.ceil( b_reduced_shots/a_total_shots )
            
            return weight_factors, R, A_shots, B_shots
            