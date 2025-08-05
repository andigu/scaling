"""Refactored simulator classes with improved architecture."""

from typing import List, Optional, Tuple
import os
import numpy as np
import stim
from multiprocessing import Pool, get_context

from ..algorithm import Algorithm
from ..codes import CodeFactory
from ..logical_circuit import LogicalCircuit
from ..noise_model import NoiseModel
from .loss_manager import LossManager
from .simulator import BaseSimulator, SampleResult
from .logical_error_calculator import LogicalErrorCalculator


def _sample_lossy_worker(args: Tuple[LogicalCircuit, int, int, int, List[int], List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Worker function for parallel lossy simulation.
    
    Args:
        args: Tuple containing:
            - circuit: LogicalCircuit
            - n_qubits: Number of qubits in the circuit
            - shots_per_config: Number of shots per loss configuration
            - seed: Random seed for this worker
            - config_indices: List of configuration indices to process
            - stim_seeds: List of stim seeds for each configuration
            
    Returns:
        Tuple of (measurements, lost_measurements) arrays
    """
    circuit, n_qubits, shots_per_config, seed, config_indices, stim_seeds = args
    
    # Initialize worker's RNG
    rng = np.random.default_rng(seed)
    
    # Process each configuration
    measurements = []
    loss_manager = LossManager(n_qubits, rng, len(config_indices))
    
    # Apply loss and get circuits for all configurations
    circs, lost_measurements = loss_manager.apply_loss(circuit)
    
    # Repeat lost measurements for each shot within a loss configuration
    lost_measurements_expanded = np.repeat(lost_measurements, shots_per_config, axis=0)
    
    for i, config_idx in enumerate(config_indices):
        # Compile and sample this configuration
        circ = stim.Circuit(circs[i]).compile_sampler(seed=stim_seeds[i])
        measurements.append(circ.sample(shots=shots_per_config))
    
    return np.concatenate(measurements, axis=0), lost_measurements_expanded


class StimSimulator(BaseSimulator):
    """
    Stim-based quantum circuit simulator for single-shot (non-layered) simulations.
    
    This simulator handles both lossy and lossless noise models. It constructs
    a complete logical circuit and runs a single simulation pass to obtain
    detector outcomes and logical errors.
    """

    def __init__(self, algorithm: Algorithm, noise_model: NoiseModel,
                 code_factory: CodeFactory, seed: int = 0):
        """
        Initializes the `StimSimulator`.

        Args:
            algorithm: The `Algorithm` instance to be simulated.
            noise_model: The `NoiseModel` to apply during simulation.
            code_factory: The `CodeFactory` to create `Code` instances for logical qubits.
            seed: The random seed for the simulator's internal random number generator.
        """
        super().__init__(algorithm, noise_model, code_factory, seed)

        # Determine if we need deterministic output signs for lossy simulation
        self.has_loss = noise_model.has_loss

        if self.has_loss:
            # Generate deterministic output signs for lossy simulation to ensure
            # that the logical error calculation is well-defined even with heralded loss.
            self.deterministic_output_signs = self.rng.choice(2, size=algorithm.n)
            lc, logical_qubits, mt = LogicalCircuit.from_algorithm(
                algorithm, code_factory, deterministic_output_signs=self.deterministic_output_signs
            )
            # Apply noise model to the circuit
            self.lossy_circuit = noise_model(lc)
            self.m2d_matrix = mt.m2d_matrix
            self.n_qubits = lc.n_qubits
        else:
            self.deterministic_output_signs = None
            lc, logical_qubits, mt = LogicalCircuit.from_algorithm(algorithm, code_factory)
            # Compile the noiseless circuit to Stim format
            self.stim_circ = noise_model(lc).to_stim()

        self.logical_qubits = logical_qubits
        self.meas_tracker = mt

    def sample(self, shots: int = 1024, n_loss_configs: int = 1, num_processes: int = 1) -> SampleResult:
        """
        Samples detector events and logical errors from the quantum circuit.
        
        Args:
            shots: The number of simulation shots to run.
            n_loss_configs: The number of distinct loss configurations to sample
                            for lossy simulations. The total `shots` will be
                            divided among these configurations. (Only relevant for lossy simulations).
            num_processes: Number of processes to use for parallel execution. 
                          Use -1 for all available CPUs. Default is 1 (no parallelization).
            
        Returns:
            A `SampleResult` object containing detector events, logical errors,
            and the measurement tracker for the simulation.
        """
        if self.has_loss:
            return self._sample_lossy(shots, n_loss_configs, num_processes)
        else:
            return self._sample_lossless(shots)

    def _sample_lossless(self, shots: int) -> SampleResult:
        """
        Performs lossless simulation using Stim's `FlipSimulator`.

        Args:
            shots: The number of simulation shots.

        Returns:
            A `SampleResult` object with detector outcomes and logical errors.
            `detectors` shape: `(shots, num_detectors)`
            `logical_errors` shape: `(shots, num_logical_qubits)`
        """
        fs = stim.FlipSimulator(
            batch_size=shots,
            disable_stabilizer_randomization=True,
            seed=self.rng.choice(self.MAX_SEED_VALUE)
        )
        fs.do(self.stim_circ)
        result = fs.to_numpy(
            output_xs=True, output_zs=True, output_measure_flips=False,
            output_detector_flips=True, transpose=True
        )
        assert result is not None, "to_numpy should return a tuple when output flags are True"
        x_err, z_err, _, det, _ = result

        logical_errors = LogicalErrorCalculator.from_physical_errors(
            self.algorithm.get_measurement_basis(), x_err, z_err, self.logical_qubits
        )

        return SampleResult(
            measurement_tracker=self.meas_tracker,
            detectors=det,
            logical_errors=logical_errors,
            additional_data={}
        )

    def _sample_lossy(self, shots: int, n_loss_configs: int, num_processes: int = 1) -> SampleResult:
        """
        Performs lossy simulation by sampling multiple loss configurations.

        Args:
            shots: The total number of simulation shots.
            n_loss_configs: The number of distinct loss configurations.
            num_processes: Number of processes to use for parallel execution.

        Returns:
            A `SampleResult` object with detector outcomes, logical errors,
            and additional data including loss masks.
            `detectors` shape: `(shots, num_detectors)`
            `logical_errors` shape: `(shots, num_logical_qubits)`
            `additional_data['loss']` shape: `(shots, num_measurements)`
        """
        assert shots % n_loss_configs == 0, 'shots must be divisible by n_loss_configs'

        # Handle num_processes parameter
        if num_processes == -1:
            num_processes = os.cpu_count() or 1
        
        # Determine whether to use parallel processing
        use_parallel = (num_processes > 1 and n_loss_configs > 1)
        
        if use_parallel:
            # Parallel implementation
            # Pre-generate all RNG seeds
            stim_seeds = self.rng.choice(self.MAX_SEED_VALUE, size=(n_loss_configs,))
            worker_seeds = self.rng.choice(self.MAX_SEED_VALUE, size=(num_processes,))
            
            # Serialize circuit once
            circuit = self.lossy_circuit
            
            # Distribute configurations across workers
            configs_per_worker = n_loss_configs // num_processes
            remainder = n_loss_configs % num_processes
            
            # Build work distribution
            worker_args = []
            config_start = 0
            for worker_id in range(num_processes):
                # Distribute remainder evenly among first workers
                n_configs = configs_per_worker + (1 if worker_id < remainder else 0)
                if n_configs == 0:
                    continue
                    
                config_indices = list(range(config_start, config_start + n_configs))
                worker_stim_seeds = stim_seeds[config_start:config_start + n_configs].tolist()
                
                worker_args.append((
                    circuit,
                    self.n_qubits,
                    shots // n_loss_configs,
                    int(worker_seeds[worker_id]),
                    config_indices,
                    worker_stim_seeds
                ))
                config_start += n_configs
            
            # Execute parallel processing
            # Use fork on Unix for better performance, spawn on Windows
            ctx = get_context('fork') if os.name != 'nt' else get_context('spawn')
            with ctx.Pool(processes=num_processes) as pool:
                results = pool.map(_sample_lossy_worker, worker_args)
            
            # Combine results
            all_measurements = []
            all_lost_measurements = []
            for measurements, lost_measurements in results:
                all_measurements.append(measurements)
                all_lost_measurements.append(lost_measurements)
            
            measurements = np.concatenate(all_measurements, axis=0)
            lost_measurements = np.concatenate(all_lost_measurements, axis=0)
            
            # Apply lost measurement mask
            meas_sim = np.where(lost_measurements, 2, measurements)
        else:
            # Serial implementation (original code)
            stim_seed = self.rng.choice(self.MAX_SEED_VALUE, size=(n_loss_configs,))
            measurements = []
            loss_manager = LossManager(self.n_qubits, self.rng, n_loss_configs)
            # Apply loss to the circuit and get the resulting circuits and lost measurement masks
            circs, lost_measurements = loss_manager.apply_loss(self.lossy_circuit)
            # Repeat lost measurements for each shot within a loss configuration
            lost_measurements = np.repeat(lost_measurements, shots//n_loss_configs, axis=0)

            for i in range(n_loss_configs):
                # Compile and sample each lossy circuit configuration
                circ = stim.Circuit(circs[i]).compile_sampler(seed=stim_seed[i])
                measurements.append(circ.sample(shots=shots//n_loss_configs))

            # Combine measurements and lost measurement masks
            meas_sim = np.where(lost_measurements, 2, np.concatenate(measurements, axis=0))
        # Compute detectors from measurements using the m2d matrix
        det = ((self.m2d_matrix @ meas_sim.T).T % 2).astype(bool)
        if self.deterministic_output_signs is None:
            raise ValueError('Deterministic output signs are required for lossy simulation')
        logical_errors = LogicalErrorCalculator.from_measurements(
            self.algorithm, meas_sim, self.logical_qubits,
            self.meas_tracker, self.deterministic_output_signs
        )
        
        return SampleResult(
            measurement_tracker=self.meas_tracker,
            detectors=det,
            logical_errors=logical_errors,
            additional_data={'loss': lost_measurements}
        )


class LayeredStimSimulator(BaseSimulator):
    """
    Layered Stim-based quantum circuit simulator for both lossy and lossless noise models.
    
    This simulator processes quantum algorithms layer by layer, providing results
    for each time step. It automatically handles both lossy and lossless cases.
    """

    def __init__(self, algorithm: Algorithm, noise_model: NoiseModel,
                 code_factory: CodeFactory, seed: int = 0):
        """
        Initializes the `LayeredStimSimulator`.

        Args:
            algorithm: The `Algorithm` instance to be simulated.
            noise_model: The `NoiseModel` to apply during simulation.
            code_factory: The `CodeFactory` to create `Code` instances for logical qubits.
            seed: The random seed for the simulator's internal random number generator.
        """
        super().__init__(algorithm, noise_model, code_factory, seed)

        # Determine if we need deterministic output signs for lossy simulation
        self.has_loss = noise_model.has_loss

        if self.has_loss:
            # warnings.warn('LayeredSimulator will use an approximation for lossy simulation, wherein qubits'
            #               'will be reset to the maximally mixed state upon being lost (see stim\'s HERALDED_ERASURE).'
            #               'This is because keeping measurements consistent across layers, with *different* initializations'
            #               'is not possible. For exact lossy simulation, use StimSimulator.')

            # Construct layered logical circuits for lossy simulation
            (init, compute, readout), qubits, mts = LogicalCircuit.from_algorithm_layered(
                algorithm, code_factory
            )
            self.init = init[0].to_stim() # Initial state preparation circuit
            # Apply noise model to compute and readout circuits for each layer
            self.compute_circuits = [noise_model(compute_circuit) for compute_circuit in compute]
            self.readout_circuits = [noise_model(readout_circuit) for readout_circuit in readout]
        else:
            # Construct layered logical circuits for lossless simulation
            (init, compute, readout), qubits, mts = LogicalCircuit.from_algorithm_layered(
                algorithm, code_factory
            )
            self.init = init[0].to_stim()  # Noiseless init circuit
            # Compile noiseless compute and readout circuits to Stim format
            self.compute_circuits_stim = [noise_model(compute_circuit).to_stim() for compute_circuit in compute]
            self.readout_circuits_stim = [noise_model(readout_circuit).to_stim() for readout_circuit in readout]

        self.logical_qubits = qubits
        self.meas_trackers = mts
        # Determine the total number of physical qubits
        self.n_qubits = max([init[0].n_qubits, compute[0].n_qubits, readout[0].n_qubits])
        self.has_loss = noise_model.has_loss

    def sample(self, shots: int = 1024) -> List[SampleResult]:
        """
        Samples detector events and logical errors for each layer of the algorithm.
        
        Args:
            shots: The number of simulation shots to run for each layer.
            
        Returns:
            A list of `SampleResult` objects, one for each layer of the algorithm.
            Each `SampleResult` contains:
            - `detectors` shape: `(shots, num_detectors)`
            - `logical_errors` shape: `(shots, num_logical_qubits)`
            - `additional_data['loss']` shape (if lossy): `(shots, num_measurements_in_layer)`
        """
        if self.has_loss:
            return self._sample_lossy(shots)
        else:
            return self._sample_lossless(shots)

    def _sample_lossless(self, shots: int) -> List[SampleResult]:
        """
        Performs layered lossless simulation using Stim's `FlipSimulator`.

        Args:
            shots: The number of simulation shots.

        Returns:
            A list of `SampleResult` objects, one for each layer.
            `detectors` shape: `(shots, num_detectors)`
            `logical_errors` shape: `(shots, num_logical_qubits)`
        """
        fs = stim.FlipSimulator(
            batch_size=shots,
            disable_stabilizer_randomization=True,
            seed=self.rng.choice(self.MAX_SEED_VALUE)
        )
        fs.do(self.init) # Initialize the simulator state

        results = []
        # Iterate through compute and readout circuits for each layer
        for compute, readout in zip(self.compute_circuits_stim, self.readout_circuits_stim):
            fs.do(compute) # Apply compute circuit for the current layer

            fs_terminal = fs.copy() # Create a copy to perform readout without affecting the main simulator state
            fs_terminal.do(readout) # Apply readout circuit
            
            # Extract results (physical errors, detectors)
            result = fs_terminal.to_numpy(
                output_xs=True, output_zs=True, output_measure_flips=False,
                output_detector_flips=True, transpose=True
            )
            assert result is not None, "to_numpy should return a tuple when output flags are True"
            x_err, z_err, _, det, _ = result
            # Calculate logical errors
            logical_errors = LogicalErrorCalculator.from_physical_errors(
                self.algorithm.get_measurement_basis(), x_err, z_err, self.logical_qubits
            )

            results.append(SampleResult(
                measurement_tracker=self.meas_trackers[len(results)], # Get the appropriate measurement tracker for this layer
                detectors=det,
                logical_errors=logical_errors,
                additional_data={}
            ))

        return results

    def _sample_lossy(self, shots: int) -> List[SampleResult]:
        """
        Performs layered lossy simulation using an approximation where lost qubits
        are reset to the maximally mixed state.

        Args:
            shots: The number of simulation shots.

        Returns:
            A list of `SampleResult` objects, one for each layer.
            `detectors` shape: `(shots, num_detectors)`
            `logical_errors` shape: `(shots, num_logical_qubits)`
            `additional_data['loss']` shape: `(shots, cumulative_num_measurements_up_to_layer)`
        """
        cumulative_meas_mask = np.zeros((shots, 0), dtype=bool)
        results = []
        stim_seed = self.rng.choice(self.MAX_SEED_VALUE)
        fs = stim.FlipSimulator(
            batch_size=shots,
            disable_stabilizer_randomization=True,
            seed=stim_seed
        )
        fs.do(self.init) # Initialize the simulator state
        
        loss_manager = LossManager(self.n_qubits, self.rng, shots)
        for i in range(len(self.compute_circuits)):
            # Apply compute circuit with loss approximation
            fs, meas_mask = loss_manager.loss_approximation(fs, self.compute_circuits[i])
            fs_readout = fs.copy() # Create copy for readout
            cumulative_meas_mask = np.concatenate([cumulative_meas_mask, meas_mask], axis=1) # Accumulate loss masks
            # Apply readout circuit with loss approximation
            fs_readout, readout_mask = loss_manager.loss_approximation(fs_readout, self.readout_circuits[i])

            # Extract results and calculate logical errors
            result = fs_readout.to_numpy(
                output_xs=True, output_zs=True, output_measure_flips=False,
                output_detector_flips=True, transpose=True
            )
            assert result is not None, "to_numpy should return a tuple when output flags are True"
            x_err, z_err, _, det, _ = result
            logical_errors = LogicalErrorCalculator.from_physical_errors(
                self.algorithm.get_measurement_basis(), x_err, z_err, self.logical_qubits
            )

            results.append(SampleResult(
                measurement_tracker=self.meas_trackers[i],
                detectors=det,
                logical_errors=logical_errors,
                additional_data={'loss': np.concatenate([cumulative_meas_mask, readout_mask], axis=1)}
            ))
        return results
