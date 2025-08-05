# Commenting out for now, galois has dependency issues

# from ast import Dict
# from collections.abc import Iterable
# from functools import lru_cache
# import numpy as np
# import pandas as pd
# import galois
# from typing import Tuple, Dict, Any
# from collections import OrderedDict

# from ..types import Pauli
# from ..instruction import Instruction
# from ..measurement_tracker import SyndromeType
# from .css import CSSCode
# from .decorators import supports_transversal_gates
# from .types import SyndromeExtractionProcedure, CodeMetadata

# @supports_transversal_gates(gates_1q=['I', 'X', 'Z'], gates_2q=['CX'])
# class BivariateBicycle(CSSCode):
#     NAME = 'BivariateBicycle'

#     @staticmethod
#     def allowable_code_params() -> Iterable[Dict[str, Any]]:
#         return [
#             {'l': 6, 'm': 6},
#             {'l': 15, 'm': 3},
#             {'l': 9, 'm': 6},
#             {'l': 12, 'm': 6},
#             {'l': 12, 'm': 12},
#             {'l': 30, 'm': 6},
#             {'l': 21, 'm': 18},
#         ]

#     @staticmethod
#     def create_metadata(**code_params) -> CodeMetadata:
#         l, m = int(code_params['l']), int(code_params['m'])
#         data = pd.DataFrame({
#             'data_id': np.arange(2*m*l),
#             'group': np.concatenate([np.zeros(m*l), np.ones(m*l)]),
#         })
        
#         x = np.kron(np.roll(np.eye(l), 1, axis=1), np.eye(m))
#         y = np.kron(np.eye(l), np.roll(np.eye(m), 1, axis=1))
        
#         A = x@x@x + 2*y + 3*y@y # Coefficients just for IDs
#         B = y@y@y + 2*x + 3*x@x
#         B = np.where(B != 0, B+3, B)
        
#         X_check = np.concatenate([A, B], axis=1) #(n/2 by n)
#         Z_check = np.concatenate([B.T, A.T], axis=1)
#         Z_check = np.where(Z_check != 0, Z_check+6, Z_check)

#         for basis, checks in zip(['X', 'Z'], [X_check, Z_check]):
#             i, j = np.argwhere(checks).T
#             edge_type = checks[i, j] - 1
#             edge_pauli = Pauli.X if basis == 'X' else Pauli.Z # X check
#             check_data_i = pd.DataFrame({
#                 'check_id': (i + len(X_check)) if basis == 'Z' else i,
#                 'data_id': j,
#                 'edge_type': edge_type.astype(int),
#                 'pauli': edge_pauli
#             })
#             if basis == 'X':
#                 x_check_data = check_data_i
#             else:
#                 z_check_data = check_data_i
#         check_data = pd.concat([x_check_data, z_check_data]).reset_index(drop=True)
#         check = pd.DataFrame({
#             'check_id': np.arange(2*m*l),
#             'check_type': np.concatenate([np.full(m*l, Pauli.X), np.full(m*l, Pauli.Z)]),
#         })
        
#         # Convert to GF2 for linear algebra operations
#         X_check_gf2 = galois.GF2((X_check != 0).astype(int))
#         Z_check_gf2 = galois.GF2((Z_check != 0).astype(int))
        
#         if not np.all(X_check_gf2 @ Z_check_gf2.T == 0):
#             raise ValueError("CSS condition violated")
        
#         X_logicals = _find_logical_operators(X_check_gf2, Z_check_gf2.null_space())
#         Z_logicals = _find_logical_operators(Z_check_gf2, X_check_gf2.null_space())
        
#         X_logicals, Z_logicals = _canonicalize_logicals(X_logicals, Z_logicals)
#         _validate_logical_operators(X_check_gf2, Z_check_gf2, X_logicals, Z_logicals)

#         i, j = np.argwhere(X_logicals).T
#         obs = pd.DataFrame({
#             'logical_id': i,
#             'logical_pauli': Pauli.X, 
#             'data_id': j,
#             'physical_pauli': Pauli.X
#         })
#         i, j = np.argwhere(Z_logicals).T
#         obs = pd.concat([obs, pd.DataFrame({
#             'logical_id': i,
#             'logical_pauli': Pauli.Z, 
#             'data_id': j,
#             'physical_pauli': Pauli.Z
#         })]).reset_index(drop=True)
#         return CodeMetadata(
#             tanner=check_data.to_records(index=False),
#             logical_operators=obs.to_records(index=False),
#             data=data.to_records(index=False),
#             check=check.to_records(index=False),
#         )
    
#     @staticmethod
#     def get_qubit_ids(index_start: int = 0, **code_params) -> OrderedDict[str, np.ndarray]:
#         l, m = code_params['l'], code_params['m']
#         return OrderedDict([
#             ('data', np.arange(index_start, index_start + 2*l*m)),
#             ('check', np.arange(index_start + 2*l*m, index_start + 4*l*m)),
#         ])

#     @lru_cache
#     @staticmethod
#     def get_syndrome_extraction_instructions(l: int, m: int) -> SyndromeExtractionProcedure:
#         ret = []
#         metadata = BivariateBicycle.create_metadata(l=l, m=m)
#         qubit_ids = BivariateBicycle.get_qubit_ids(l=l, m=m)
#         data_ids, check_ids = qubit_ids['data'], qubit_ids['check']

#         data, check, tanner = metadata.data, metadata.check, metadata.tanner
#         x_check_id = check_ids[check[check['check_type'] == Pauli.X]['check_id']]
#         z_check_id = check_ids[check[check['check_type'] == Pauli.Z]['check_id']]

#         ret.append(Instruction('RX', x_check_id)) # group 0 is X check
#         ret.append(Instruction('R', z_check_id)) # group 1 is Z check
        
#         # A1: 0, A2: 1, A3: 2, B1: 3, B2: 4, B3: 5, A1^T: 6, A2^T: 7, A3^T: 8, B1^T: 9, B2^T: 10, B3^T: 11
#         edge_type_by_name = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'A1^T', 'A2^T', 'A3^T', 'B1^T', 'B2^T', 'B3^T']
#         ordering = ['A1^T', 'A2', 'A3^T', 'B2', 'B1^T', 'B1', 'B2^T', 'B3', 'B3^T', 'A1', 'A2^T', 'A3']
#         for name in ordering:
#             edge_type = edge_type_by_name.index(name)
#             grp = tanner[tanner['edge_type'] == edge_type]
#             if edge_type < 6:
#                 ret.append(Instruction('CX', np.stack([check_ids[grp['check_id']], data_ids[grp['data_id']]], axis=1)))
#             else:
#                 ret.append(Instruction('CX', np.stack([data_ids[grp['data_id']], check_ids[grp['check_id']]], axis=1)))
        
#         ret.append(Instruction('MX', x_check_id))
#         ret.append(Instruction('M', z_check_id))
        
#         return SyndromeExtractionProcedure(
#             instructions=ret,
#             syndrome_ids=[check[check['check_type'] == Pauli.X]['check_id'].tolist(), 
#                           check[check['check_type'] == Pauli.Z]['check_id'].tolist()],
#             syndrome_types=[SyndromeType.CHECK, SyndromeType.CHECK],
#         )

# def _find_pivots(reduced_matrix: np.ndarray) -> np.ndarray:
#     """Find pivot columns in a row-reduced matrix."""
#     pivots = []
#     for row in range(len(reduced_matrix)):
#         if any(reduced_matrix[row]):
#             pivots.append(np.flatnonzero(reduced_matrix[row]).min())
#     return np.unique(pivots)


# def _find_logical_operators(check_matrix: galois.FieldArray, null_space: galois.FieldArray) -> galois.FieldArray:
#     """Extract logical operators from check matrix null space."""
#     # Combine check matrix transpose with null space vectors
#     combined = np.concatenate([check_matrix.T, null_space.T], axis=1).view(galois.GF2)
    
#     # Row reduce to identify independent logical operators
#     rref = combined.row_reduce(eye='left').view(np.ndarray)
    
#     # Find pivot columns corresponding to logical operators
#     pivots = _find_pivots(rref)
#     pivots = pivots - check_matrix.shape[0]  # Adjust for check matrix columns
#     pivots = pivots[pivots >= 0]  # Keep only logical operator indices
    
#     return null_space[pivots].view(galois.GF2)


# def _canonicalize_logicals(X_logicals: galois.FieldArray, Z_logicals: galois.FieldArray) -> Tuple[galois.FieldArray, galois.FieldArray]:
#     """Put logical operators in canonical form where [X_i, Z_j] = δ_ij."""
#     n_logical = len(X_logicals)
    
#     commutator_matrix = X_logicals @ Z_logicals.T
#     augmented_matrix = np.concatenate([commutator_matrix, np.eye(n_logical, dtype=int)], axis=1)
#     reduced_matrix = augmented_matrix.view(galois.GF2).row_reduce()
    
#     transformation = reduced_matrix[:, n_logical:]
#     X_logicals_canonical = transformation @ X_logicals
    
#     return X_logicals_canonical.view(galois.GF2), Z_logicals.view(galois.GF2)


# def _validate_logical_operators(X_check: galois.FieldArray, Z_check: galois.FieldArray, 
#                                X_logicals: galois.FieldArray, Z_logicals: galois.FieldArray) -> None:
#     if np.any((Z_check @ X_logicals.T) != 0) or np.any((X_check @ Z_logicals.T) != 0):
#         raise ValueError('Logical operators do not commute with stabilizer checks')
#     n_logical = len(X_logicals)
#     if np.any(Z_logicals @ X_logicals.T != np.eye(n_logical, dtype=int)):
#         raise ValueError('Logical operators do not satisfy canonical commutation relations')
        
