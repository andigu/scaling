# Quantum Error Correction Codes

This module provides implementations of quantum error correction codes with a unified interface for syndrome extraction, logical operations, and resource management. The codes are designed to integrate seamlessly with the simulation framework and neural decoder pipeline.

## Architecture Overview

The code architecture follows a hierarchical design:

- **`Code`**: Abstract base class defining the QEC interface
- **`CSSCode`**: Base class for CSS (Calderbank-Shor-Steane) codes
- **Concrete Implementations**: `SurfaceCode`, `ColorCode`, `SteaneSurfaceCode`, `QRMCode`

## Available Codes

### Surface Codes (`surface.py`)

Surface codes are topological quantum error correction codes defined on 2D square lattices. They provide excellent error correction performance with local stabilizer checks.

#### Surface Code Variants

```python
from simulate.codes import SurfaceCode, SteaneSurfaceCode

# Standard surface code with ancilla-based syndrome extraction
surface_code = SurfaceCode.from_code_params(d=5, index_start=0, block_id=0)

# Steane surface code with flag-based error correction
steane_surface = SteaneSurfaceCode.from_code_params(d=5, index_start=0, block_id=0)
```

**Key Features:**
- **Distance Parameter**: Controls error correction capability (d=3,5,7,...)
- **Transversal Gates**: Supports H, X, Z, CX, SWAP operations
- **Hook Error Suppression**: Mitigates measurement scheduling errors
- **2D Lattice Geometry**: Data qubits on vertices, stabilizers on faces/edges

**Geometric Layout:**
- **Data Qubits**: Arranged on lattice vertices  
- **X-Checks**: Measure X-type stabilizers (star operators)
- **Z-Checks**: Measure Z-type stabilizers (plaquette operators)
- **Boundary Conditions**: Rough/smooth boundaries for logical operators

### Color Codes (`color.py`)

Color codes are topological codes defined on honeycomb lattices with enhanced transversal gate sets, particularly useful for universal quantum computation.

```python
from simulate.codes import ColorCode

# Triangular color code
color_code = ColorCode.from_code_params(d=3, index_start=0, block_id=0)
```

**Key Features:**
- **Enhanced Gate Set**: Supports H, S, S_DAG, X, Y, Z, CX, SWAP transversally
- **Triangular Lattice**: Defined on honeycomb/triangular graph structure  
- **Three-Colorable**: Faces colored with three colors (red, green, blue)
- **CSS Structure**: X and Z stabilizers on different colored faces

**Geometric Properties:**
- **Face-Based Stabilizers**: Each face corresponds to a stabilizer check
- **Graph Representation**: Uses graph neural networks for ML processing
- **Angular Relationships**: 6-fold rotational symmetry in face connections

### Usage Patterns

#### Creating Code Instances

```python
from simulate.codes import SurfaceCode, ColorCode, CodeFactory

# Direct instantiation (manual resource management)
code = SurfaceCode.from_code_params(
    d=5,                    # Distance parameter
    index_start=0,          # Starting physical qubit index
    block_id=0      # Logical qubit identifier
)

# Factory pattern (automatic resource management)
factory = CodeFactory(SurfaceCode, {'d': 5})
code1 = factory()  # Gets qubits 0-48, logical_id=0
code2 = factory()  # Gets qubits 49-97, logical_id=1
```

#### Syndrome Extraction

```python
from simulate import MeasurementTracker

# Get syndrome extraction instructions
tracker = MeasurementTracker()
syndrome_data = code.get_syndrome_extraction_instructions()

# Access structured syndrome data
instructions = syndrome_data.instructions  # List[Instruction]
syndrome_ids = syndrome_data.syndrome_ids   # List[np.ndarray]
syndrome_types = syndrome_data.syndrome_types # List[SyndromeType]
```

#### Logical Operations

```python
# Logical Pauli operations (identity on other codes)
instructions = code.apply_gate('X', target=None)   # Logical X
instructions = code.apply_gate('Z', target=None)   # Logical Z

# Two-qubit logical operations
instructions = code.apply_gate('CX', target=other_code)  # Logical CNOT

# Transversal Hadamard (requires qubit reordering)
instructions = code.apply_gate('H', target=None)
```

#### Data Measurement

```python
# Measure data qubits in computational basis
z_measurements = code.measure_data(basis='Z')

# Measure data qubits in Hadamard basis  
x_measurements = code.measure_data(basis='X')
```

## Code Interface (`code.py`)

### Core Abstract Methods

Every quantum error correction code must implement:

```python
class MyCode(Code):
    def get_syndrome_extraction_instructions(self, block_id: int) -> SyndromeExtractionData:
        """Return instructions for syndrome extraction."""
        pass
    
    def measure_data(self, basis: str = 'Z', **kwargs) -> List[Instruction]:
        """Return instructions for data qubit measurement."""
        pass
    
    def observable(self, obs: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return logical observable operator (Pauli strings, qubit indices)."""
        pass
```

### Transversal Gate Support

The `@supports_transversal_gates` decorator automates transversal gate validation:

```python
@supports_transversal_gates(
    gates_1q=['H', 'X', 'Z'],  # Single-qubit transversal gates
    gates_2q=['CX', 'SWAP']    # Two-qubit transversal gates  
)
class MyCode(Code):
    # Must implement _apply_h, _h_detectors, _apply_x, _x_detectors, etc.
    def _apply_h(self, **kwargs) -> List[Instruction]:
        """Implement logical Hadamard gate."""
        pass
        
    def _h_detectors(self, meas_track: MeasurementTracker, time: int):
        """Add detectors for Hadamard gate."""
        pass
```

This design enforces compile-time validation that all claimed transversal gates have proper implementations.

### CSS Code Base Class

For CSS codes, inherit from `CSSCode` for automatic CNOT implementation:

```python
class MyCSSCode(CSSCode):
    # Inherits logical CNOT implementation
    # Must implement syndrome extraction and logical observables
    pass
```

## Resource Management

### Physical Qubit Allocation

The framework manages physical qubit indices automatically:

```python
# Factory handles resource allocation
factory = CodeFactory(SurfaceCode, {'d': 3})

code1 = factory()  # Uses qubits 0-16
code2 = factory()  # Uses qubits 17-33  
code3 = factory()  # Uses qubits 34-50

# Manual allocation for fine control
code = SurfaceCode.from_code_params(
    d=3,
    index_start=100,      # Start at qubit 100
    block_id=5    # Logical qubit ID 5
)
```

### Metadata Access

Each code provides structured metadata for ML processing:

```python
# Access qubit organization
data_qubits = code.qubit_ids['data']      # Data qubit indices
check_qubits = code.qubit_ids['check']    # Check qubit indices

# Access geometric information
positions = code.metadata['data']['pos_x'], code.metadata['data']['pos_y']
check_types = code.metadata['check']['type']  # 'X' or 'Z'

# Code parameters
distance = code.code_params['d']
total_qubits = code.num_qubits
logical_qubits = code.n_logical_qubits
```

## Integration Points

### With Simulation Framework

```python
from simulate import LogicalCircuit, Algorithm

# Codes integrate directly with circuit construction
algorithm = Algorithm.build_random(n_qubits=2, depth=10)
logical_circuit, logical_qubits, tracker = LogicalCircuit.from_algorithm(
    algorithm=algorithm,
    code=surface_code,
    n_rounds=10
)
```

### With Neural Decoders

```python
# Codes provide structured data for ML preprocessing
syndrome_data = code.get_syndrome_extraction_instructions(0)

# Metadata used by ML embedders
positions = code.metadata['check']['pos_x'], code.metadata['check']['pos_y']
check_types = code.metadata['check']['type']

# Used in graph neural networks (ColorCode) 
face_graph = code.get_face_graph()  # For ColorCode only
```

## Visualization (`visualization.py`)

Debugging and analysis utilities:

```python
from simulate.codes import Visualizer

# Visualize code layout
vis = Visualizer(code)
vis.plot_lattice()           # Show qubit positions and connectivity
vis.plot_syndrome(syndrome)  # Highlight syndrome measurements
vis.animate_error_propagation(errors)  # Show error evolution
```

## Extension Guide

### Implementing a New Code

1. **Choose Base Class**: `Code` or `CSSCode`
2. **Define Geometry**: Implement qubit layout and metadata
3. **Syndrome Extraction**: Implement stabilizer measurement protocol
4. **Logical Operators**: Define logical X/Z observables
5. **Transversal Gates**: Use decorator and implement gate methods

```python
@supports_transversal_gates(gates_1q=['X', 'Z'])
class MyNewCode(CSSCode, LatticeCode):
    NAME = 'MyNewCode'
    
    @classmethod
    def from_code_params(cls, d: int, **kwargs):
        # Create qubit layout and metadata
        qubit_ids = OrderedDict([
            ('data', np.array([...])),
            ('check', np.array([...]))
        ])
        metadata = {...}
        return cls(qubit_ids, metadata, d=d, **kwargs)
    
    def get_syndrome_extraction_instructions(self, block_id):
        # Implement syndrome measurement protocol
        pass
    
    def observable(self, obs: str):
        # Define logical observables
        pass
```

## Performance Considerations

- **Lazy Initialization**: Metadata computed on-demand
- **NumPy Arrays**: Efficient qubit indexing and operations  
- **Graph Caching**: Pre-computed connectivity
- **Memory Efficiency**: Structured arrays for large-scale simulation

## See Also

- **Simulation Framework**: `src/simulate/README.md`
- **Simulation Engines**: `src/simulate/simulator/README.md`  
- **ML Integration**: `src/ml/README.md`
