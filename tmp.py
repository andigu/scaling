from torch_geometric.nn import CuGraphRGCNConv
from torch_geometric import EdgeIndex
import torch

batch_size, num_nodes, embedding_dim = 16, 10, 32
rgcn = CuGraphRGCNConv(embedding_dim, embedding_dim, num_relations=12).to('cuda')
X = torch.randn(batch_size, num_nodes, embedding_dim, device='cuda', dtype=torch.float32)
edges = torch.randint(0, num_nodes, size=(2, 5), device='cuda')
edge_types = torch.randint(0, 12, size=(5, ), device='cuda')
print(rgcn(X, EdgeIndex(edges), edge_types))