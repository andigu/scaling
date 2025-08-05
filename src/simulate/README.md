# Quantum Error Correction Simulation Framework

This module provides a comprehensive framework for simulating quantum error correction codes with realistic noise models. The simulation architecture is designed to support various quantum error correction codes and enable integration with machine learning decoders.

## Core Architecture

The simulation framework follows a layered architecture that separates concerns:

**Algorithm Definition** → **Code Implementation** → **Circuit Construction** → **Simulation** → **Measurement Tracking**

### Key Components

#### `algorithm.py` - Quantum Algorithm Representation
Represents quantum algorithms as sequences of gate layers:

```python
import numpy as np
from simulate import Algorithm, Layer

# Create a quantum algorithm with layers of gates
algorithm = Algorithm([
    Layer({"H": [0], "CX": [[0, 1]]}),  # Hadamard on qubit 0, CNOT on qubits 0,1
    Layer({"Z": [1], "X": [0]})         # Pauli gates
])

# Generate random quantum circuits
random_algorithm = Algorithm.build_random(
    n=3,
    depth=10,
    rng=np.random.default_rng(),
    single_qubit_gates=["H", "Z", "X", "Y", "S"],
    entangle_prob=0.1
)
```

The `Layer` class ensures that each qubit can only have one gate per layer, preventing conflicting gate assignments.

#### `codes/` - Quantum Error Correction Codes
Modular implementations of quantum error correction codes:

- **`base.py`**: Abstract base class defining the QEC interface
- **`surface.py`**: Surface code implementation (including Steane variant)
- **`color.py`**: Color code implementation
- **`qrm.py`**: QRM code implementation
- **`bb.py`**: BB code implementation (experimental)
- **`visualization.py`**: Plotting and debugging utilities

```python
from simulate import SurfaceCode, ColorCode, SteaneSurfaceCode, QRMCode

# Create different QEC codes
surface_code = SurfaceCode.from_code_params(d=5)
color_code = ColorCode.from_code_params(d=3)
steane_surface = SteaneSurfaceCode.from_code_params(d=5)
qrm_code = QRMCode.from_code_params(d=5)
```

#### `logical_circuit.py` - Circuit Construction
Converts high-level algorithms combined with error correction codes into executable quantum circuits:

```python
import numpy as np
from simulate import LogicalCircuit, Algorithm, SurfaceCode

algorithm = Algorithm.build_random(n=2, depth=5, rng=np.random.default_rng(), 
                                   single_qubit_gates=['H', 'X', 'Y', 'Z'])
code = SurfaceCode.from_code_params(d=3)

# Build fault-tolerant implementation
logical_circuit, logical_qubits, measurement_tracker = LogicalCircuit.from_algorithm(
    algorithm=algorithm,
    code=code,
    n_rounds=10
)
```

#### `measurement_tracker.py` - Central Measurement Coordination
The `MeasurementTracker` serves as the critical bridge between simulation and machine learning:

```python
from simulate import MeasurementTracker, MeasurementRole

# Measurement roles across all QEC codes
roles = [
    MeasurementRole.X_CHECK,  # X-type stabilizer measurement
    MeasurementRole.Z_CHECK,  # Z-type stabilizer measurement  
    MeasurementRole.X_FLAG,   # X-type flag qubit
    MeasurementRole.Z_FLAG,   # Z-type flag qubit
    MeasurementRole.DATA_X,   # Data qubit X measurement
    MeasurementRole.DATA_Z    # Data qubit Z measurement
]
```

The tracker uses structured NumPy arrays to organize measurement data:
- **Spatial-temporal indexing**: Each measurement has position, time, and type
- **Logical error calculation**: Computes whether logical errors occurred
- **ML integration**: Provides structured syndrome data for neural decoders

#### `simulator/` - Simulation Engines
Realistic quantum simulation with noise models:

- **`stim_simulators.py`**: Interface to the Stim quantum circuit simulator
- **`simulator.py`**: Base classes and result structures
- **`logical_error_calculator.py`**: Determines logical error outcomes
- **`loss_manager.py`**: Handles lossy quantum channels

```python
from simulate import LayeredStimSimulator, NoiseModel

# Configure realistic noise
noise_model = NoiseModel(
    single_qubit_error_rate=1e-3,
    two_qubit_error_rate=1e-2,
    measurement_error_rate=1e-3
)

# Run simulation
simulator = LayeredStimSimulator(algorithm, noise_model, code_factory)
result = simulator.sample(shots=1000)
```

#### `noise_model.py` - Realistic Error Models
Decorates a circuit with the correct noise instructions:
- **Depolarizing noise**: Random Pauli errors on gates
- **Measurement errors**: Bit-flip errors on measurement outcomes
- **Loss channels**: Physical qubit loss during computation

## Data Flow

1. **Algorithm Creation**: Define quantum circuits as gate sequences
2. **Code Selection**: Choose appropriate quantum error correction code
3. **Circuit Construction**: Combine algorithm + code → fault-tolerant circuit
4. **Noise Application**: Add realistic error models to operations
5. **Simulation**: Execute circuit using Stim simulator
6. **Syndrome Extraction**: Track stabilizer measurements via MeasurementTracker
7. **Logical Error Detection**: Determine if logical errors occurred

## Integration with Machine Learning

The simulation framework is designed to seamlessly integrate with neural decoders:

- **Structured Data**: MeasurementTracker provides ML-friendly data formats
- **On-the-fly Generation**: Infinite dataset creation from random algorithms
- **Flexible Parameters**: Vary code distances, noise levels, and circuit complexity
- **Detector Coordination**: Consistent syndrome measurement indexing

## Usage Examples

### Basic Simulation Workflow
```python
import numpy as np
from simulate import Algorithm, CodeFactory, SurfaceCode, LayeredStimSimulator, NoiseModel

# 1. Create components
algorithm = Algorithm.build_random(n=1, depth=20, rng=np.random.default_rng(), 
                                   single_qubit_gates=['H', 'X', 'Y', 'Z'])
code_factory = CodeFactory(SurfaceCode, {'d': 5})
noise_model = NoiseModel(single_qubit_error_rate=1e-3)

# 2. Run simulation
simulator = LayeredStimSimulator(algorithm, noise_model, code_factory)
results = simulator.sample(shots=1000)

# 3. Access results
for result in results:
    print(f"Logical error rate: {result.logical_errors.mean()}")
    print(f"Detector outcomes: {result.detectors.shape}")
```

### Measurement Tracking
```python
from simulate import LogicalCircuit, MeasurementTracker

# Build circuit with measurement tracking
logical_circuit, logical_qubits, tracker = LogicalCircuit.from_algorithm(
    algorithm=algorithm,
    code=code,
    n_rounds=10
)

# Access structured measurement data
measurements = tracker.measurements  # Structured array with position, time, type
detectors = tracker.detectors  # Detector metadata

# Check for logical errors
physical_errors = np.random.binomial(1, 0.01, size=code.n_physical_qubits)
logical_errors = tracker.physical_to_logical_error(physical_errors)
```

### Custom Code Implementation
```python
from simulate.codes import Code, SyndromeExtractionData
from simulate import Instruction

class MyCustomCode(Code):
    def get_syndrome_extraction_instructions(self, block_idx: int) -> SyndromeExtractionData:
        # Implement syndrome extraction for your code
        instructions = [
            Instruction("MZ", targets=[...]),  # Z measurements
            Instruction("MX", targets=[...])   # X measurements  
        ]
        return SyndromeExtractionData(
            instructions=instructions,
            targets=[...],
            types=["check", "check"]
        )
    
    # Implement other required methods...
```

## Extension Points

The framework is designed for extensibility:

1. **New QEC Codes**: Inherit from `Code` base class
2. **Custom Algorithms**: Use `Algorithm` and `Layer` classes
3. **Novel Noise Models**: Extend `NoiseModel` class
4. **Alternative Simulators**: Implement `BaseSimulator` interface

## Performance Considerations

- **Stim Integration**: Leverages highly optimized Stim simulator
- **Structured Arrays**: NumPy-based measurement tracking for efficiency
- **Batch Processing**: Supports large-scale syndrome sampling
- **Memory Management**: Careful handling of large circuit simulations

## See Also

- **Code Implementations**: `src/simulate/codes/README.md`
- **Simulation Engines**: `src/simulate/simulator/README.md`
- **ML Integration**: `src/ml/README.md`
