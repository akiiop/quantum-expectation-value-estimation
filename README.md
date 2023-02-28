# quantum-expectation-value-estimation

Measuring [expectation values](https://en.wikipedia.org/wiki/Expectation_value_(quantum_mechanics)) of [observables](https://en.wikipedia.org/wiki/Observable#Quantum_mechanics) is an essential task in  quantum experiments and quantum algorithms, for example, [Quantum Neural Networks](https://en.wikipedia.org/wiki/Quantum_neural_network) (QNN), the [Variational Quantum Eigensolver](https://en.wikipedia.org/wiki/Variational_quantum_eigensolver) (VQE) and the [Quantum Approximate Optimization Algorithm](https://en.wikipedia.org/wiki/Quantum_optimization_algorithms) (QAOA) . These hybrid quantum-classical algorithms are one of the leading candidates for obtaining quantum advantage in the current era. However, they may require such high precision in estimating the expectation values of observables that makes them impractical due to the many sources of noise in current quantum hardware. 

Based on [[2]](https://link.aps.org/doi/10.1103/PhysRevResearch.4.033173) we propose a hybrid quantum-classical algorithm that estimates the expectation value of observables acting on pure states. This algorithm, like the method presented in [2] is effective for concentrated states, i.e., when only a small number of amplitudes in a measurement basis are non-negligible, a property often satisfied for Molecular Hamiltonians. Moreover, this algorithm improves the one presented in [2], since it is also designed to be effective for Hamiltonian depending on Pauli strings that contain a small fixed number of $X$ and $Y$ Pauli operators, a property that molecular Hamiltonians often satisfy. By exploiting this property, the algorithm works properly with non-concentrated states. The algorithm has an efficient use of classical memory and computational time, and it does not depend on the number of Pauli strings.

Our algorithm is implemented in Qiskit and detailed in this tutorial, which provides the theoretical framework and numerical simulations to evaluate its performance.

References

[1] M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio & Patrick J. Coles. Variational quantum algorithms. Nature Reviews Physics. 3:625–644, 8 2021. ISSN 2522-5820. doi: 10.1038/s42254-021-00348-9.

[2] Kohda, Masaya and Imai, Ryosuke and Kanno, Keita and Mitarai, Kosuke and Mizukami, Wataru and Nakagawa, Yuya O. Quantum expectation-value estimation by computational basis sampling. Phys. Rev. Res, 4:033173, Sep 2022. doi: 10.1103/PhysRevResearch.4.033173.

[3] Gonthier, Jérôme F. and Radin, Maxwell D. and Buda, Corneliu and Doskocil, Eric J. and Abuan, Clena M. and Romero, Jhonathan. Measurements as a roadblock to near-term prac-tical quantum advantage in chemistry: Resource analysis. Phys. Rev. Res., 4:033154, Aug 2022. doi: 10.1103/PhysRevResearch.4.033154.

[4] Jarrod R McClean, Jonathan Romero, Ryan Babbush, and Al ́an Aspuru-Guzik. The theory of variational hybrid quantum-classical algorithms. New Journal of Physics, 18(2):023023, feb 2016. doi: 10.1088/1367-2630/18/2/ 023023.

[5] Hsin-Yuan Huang, Richard Kueng, and John Preskill. Predicting many properties of a quantum system from very few measurements. Nature Physics, 16:1050–1057, 10 2020. ISSN 1745-2473. doi: 10.1038/s41567-020-0932-7.

[6] M. D. Radin and P. Johnson, Classically-boosted variational quantum eigensolver, arXiv:2106.04755.

[7] Masaya Kohda, Ryosuke Imai, Keita Kanno, Kosuke Mitarai, Wataru Mizukami, and Yuya O. Nakagawa. Quantum expectation-value estimation by computational basis sampling. Phys. Rev. Res., 4:033173, Sep 2022. doi: 10.1103/PhysRevResearch.4.033173. URL https://link.aps.org/doi/10.1103/PhysRevResearch.4.033173.
