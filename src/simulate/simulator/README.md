# Quantum Circuit Simulation Engines

This module provides quantum circuit simulators for realistic quantum error correction simulations. The simulators interface with the Stim quantum computing simulator and handle both lossless and lossy quantum channels.

## Architecture Overview

The simulation architecture is designed around a hierarchy of simulator classes:

- **`BaseSimulator`**: Common functionality for all simulators
- **`Simulator`**: Single-shot simulation interface
- **`LayeredSimulator`**: Multi-layer simulation for time-resolved analysis
- **Concrete Implementations**: `StimSimulator`, `LayeredStimSimulator`

## Core Components

### `simulator.py` - Base Classes and Interfaces

Defines the foundational interfaces for all quantum simulators:

```python
from simulate.simulator import BaseSimulator, SampleResult

@dataclass
class SampleResult:
    measurement_tracker: MeasurementTracker  # Tracks measurements across space/time
    detectors: np.ndarray                    # Detector outcome samples
    logical_errors: np.ndarray               # Ground truth logical errors
    additional_data: Dict[str, Any]          # Simulator-specific data (e.g., losses)
```

The `SampleResult` structure provides a standardized format for simulation outputs that integrates seamlessly with the ML pipeline.

### `stim_simulators.py` - Stim Integration

Implements quantum circuit simulation using the Stim simulator:

#### `StimSimulator` - Single-shot Simulation
```python
from simulate import StimSimulator, NoiseModel, CodeFactory

# Configure noise model
noise_model = NoiseModel(
    single_qubit_error_rate=1e-3,
    two_qubit_error_rate=1e-2,
    measurement_error_rate=1e-4
)

# Create simulator
simulator = StimSimulator(
    algorithm=algorithm,
    noise_model=noise_model,
    code_factory=code_factory
)

# Run simulation
result = simulator.sample(shots=1000)
```

#### `LayeredStimSimulator` - Time-Resolved Simulation
```python
from simulate import LayeredStimSimulator

# Multi-layer simulation for temporal analysis
simulator = LayeredStimSimulator(
    algorithm=algorithm,
    noise_model=noise_model,
    code_factory=code_factory
)

# Returns List[SampleResult] - one per circuit layer
layer_results = simulator.sample(shots=1000)
```

**Key Features:**
- **Automatic Noise Handling**: Supports both lossless and lossy quantum channels
- **Detector Construction**: Automatically builds Stim detectors from measurement tracking
- **Memory Efficient**: Streams results for large-scale simulations
- **Reproducible**: Seeded random number generation

### `logical_error_calculator.py` - Ground Truth Computation

Computes logical errors from physical error patterns:

```python
from simulate.simulator import LogicalErrorCalculator

# Calculate logical errors from physical errors
logical_errors = LogicalErrorCalculator.from_physical_errors(
    measurement_basis=['Z', 'Z', 'X'],  # Final measurement basis
    x_err=x_error_patterns,             # Physical X errors
    z_err=z_error_patterns,             # Physical Z errors  
    logical_qubits=code_list            # QEC codes being simulated
)
```

This component bridges the gap between physical error models and logical error detection, providing ground truth labels for neural decoder training.

### `loss_manager.py` - Lossy Quantum Channel Simulation

Handles quantum operations with physical qubit loss:

```python
from simulate.simulator.loss_manager import LossManager

# Manages lossy quantum channels
loss_manager = LossManager(
    n_qubits=num_physical_qubits,
    rng=random_generator,
    batch_size=shots,
    current_losses=existing_losses  # Optional: pre-existing losses
)

# Approximates loss effects on circuit simulation
measurement_mask = loss_manager.loss_approximation(
    fs=stim_flip_simulator,
    circuit=logical_circuit
)
```

**Loss Handling Strategy:**
- **Probabilistic Loss**: Qubits lost with specified probability
- **Batch Processing**: Efficient handling of multiple shots
- **Cascade Effects**: Lost qubits affect subsequent measurements
- **Approximation Methods**: Balances accuracy with computational efficiency

## Simulation Workflow

### 1. Circuit Construction
```python
from simulate import LogicalCircuit, Algorithm, CodeFactory

# Build fault-tolerant circuit
logical_circuit, logical_qubits, tracker = LogicalCircuit.from_algorithm(
    algorithm=Algorithm.build_random(n=2, depth=10),
    code_factory=CodeFactory(SurfaceCode, {'d': 5})
)
```

### 2. Noise Model Configuration
```python
from simulate import NoiseModel

# Configure realistic noise parameters
noise_model = NoiseModel(
    single_qubit_error_rate=1e-3,    # Depolarizing error rate
    two_qubit_error_rate=1e-2,       # Two-qubit gate error rate
    measurement_error_rate=1e-4,     # Measurement flip probability
    qubit_loss_rate=1e-5             # Physical qubit loss rate
)
```

### 3. Simulation Execution
```python
# Run simulation with Stim
simulator = LayeredStimSimulator(
    algorithm=algorithm,
    noise_model=noise_model,
    code_factory=code_factory
)

results = simulator.sample(shots=10000)
```

### 4. Result Processing
```python
# Extract simulation data
for layer_idx, result in enumerate(results):
    # Measurement tracking data
    measurements = result.measurement_tracker.mc
    
    # Detector outcomes for ML training
    detector_samples = result.detectors  # Shape: (shots, n_detectors)
    
    # Ground truth logical errors
    logical_errors = result.logical_errors  # Shape: (shots, n_logical_qubits)
    
    # Additional simulator data
    if 'losses' in result.additional_data:
        qubit_losses = result.additional_data['losses']
```

## Integration with ML Pipeline

The simulators are designed to seamlessly integrate with neural decoder training:

### Data Generation
```python
from ml.data.data_module import QECDataset

# Dataset directly uses simulator results
dataset = QECDataset(
    code_class=SurfaceCode,
    preprocessor=preprocessor,
    config_sampler=config_sampler,
    data_config=data_config
)

# Generates batches of simulation data
for batch in dataset:
    detector_samples = batch['detectors']      # Input to neural decoder
    logical_errors = batch['logical_errors']   # Ground truth labels
    metadata = batch['metadata']               # Code structure information
```

### Syndrome Extraction
```python
# MeasurementTracker provides structured syndrome data
syndrome_metadata = extract_syndrome_metadata(
    measurement_tracker=result.measurement_tracker,
    detectors=result.detectors
)

# Formatted for neural network input
syndrome_tensor = preprocess_syndrome(syndrome_metadata)
```

## Performance Considerations

### Stim Optimization
- **Compiled Circuits**: Stim pre-compiles circuits for efficient execution
- **Batch Processing**: Vectorized operations across multiple shots
- **Memory Management**: Streams large datasets without memory overflow
- **Parallel Execution**: Multi-threaded simulation when possible

### Scalability Features
- **Layered Processing**: Process circuit layers independently
- **Memory Efficient**: Streaming results for large-scale simulations
- **Configurable Batch Sizes**: Balance memory usage vs. throughput
- **Reproducible Seeds**: Deterministic simulation for debugging

### Noise Model Efficiency
- **Channel Approximations**: Efficient approximation of complex noise models
- **Loss Tracking**: Optimized handling of lossy quantum channels
- **Error Correlation**: Efficient simulation of correlated errors

## Extending the Simulator Framework

### Custom Simulators
```python
from simulate.simulator import LayeredSimulator

class MyCustomSimulator(LayeredSimulator):
    def sample(self, shots: int, **kwargs) -> List[SampleResult]:
        # Implement custom simulation logic
        results = []
        for layer in self.algorithm:
            # Custom simulation for each layer
            result = self._simulate_layer(layer, shots)
            results.append(result)
        return results
```

### Novel Noise Models
```python
from simulate import NoiseModel

class CorrelatedNoiseModel(NoiseModel):
    def apply_noise(self, circuit, **kwargs):
        # Implement correlated noise effects
        pass
```

### Alternative Backends
```python
# Interface allows different quantum simulation backends
class QiskitSimulator(BaseSimulator):
    def sample(self, shots: int, **kwargs) -> SampleResult:
        # Implement Qiskit-based simulation
        pass
```

## Error Handling and Debugging

### Simulation Validation
```python
# Validate simulation results
assert result.detectors.shape[0] == shots
assert result.logical_errors.shape[1] == algorithm.n_qubits
assert len(result.measurement_tracker.mc) > 0
```

### Debugging Utilities
```python
# Access intermediate simulation data
measurement_data = result.measurement_tracker.mc
detector_metadata = result.measurement_tracker.detectors

# Visualize syndrome patterns
from simulate.codes import Visualizer
vis = Visualizer(code)
vis.plot_syndrome(result.detectors[0])  # First shot's syndrome
```

## See Also

- **Simulation Framework**: `src/simulate/README.md`
- **Error Correction Codes**: `src/simulate/codes/README.md`
- **ML Integration**: `src/ml/README.md`
- **Stim Documentation**: [Stim GitHub Repository](https://github.com/quantumlib/Stim)
