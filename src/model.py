import torch
import torch.nn as nn
import torch_geometric.nn.conv as conv
from torch_geometric.nn import BatchNorm
import math

class Decoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.p_embedder = nn.Sequential(
            nn.Linear(1, 2*embedding_dim),
            nn.GELU(),
            nn.Linear(2*embedding_dim, embedding_dim),
        )
        self.det_value_emb = nn.Embedding(2, embedding_dim)
        self.position_emb = nn.Embedding(24, embedding_dim)
        self.time_emb = nn.Embedding(11, embedding_dim)      # 4 time steps (0-3)
        self.num_layers = 3
        self.det_to_error = nn.ModuleList([
            # conv.GATv2Conv(embedding_dim, embedding_dim, heads=4, concat=False, residual=True) for _ in range(self.num_layers)
            conv.SAGEConv(embedding_dim, embedding_dim, normalize=True, project=True) for _ in range(self.num_layers)
        ])
        self.error_to_det = nn.ModuleList([
            # conv.GATv2Conv(embedding_dim, embedding_dim, heads=4, concat=False, residual=True) for _ in range(self.num_layers)
            conv.SAGEConv(embedding_dim, embedding_dim, normalize=True, project=True) for _ in range(self.num_layers)
        ])
        
        # Batch normalization for nodes after each layer
        self.error_norms = nn.ModuleList([
            BatchNorm(embedding_dim) for _ in range(self.num_layers)
        ])
        self.det_norms = nn.ModuleList([
            BatchNorm(embedding_dim) for _ in range(self.num_layers)
        ])
        
        # self.final_det_to_err = conv.GATv2Conv(embedding_dim, embedding_dim, heads=4, concat=False, residual=True)
        self.final_det_to_err = conv.SAGEConv(embedding_dim, embedding_dim, normalize=True, project=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 2*embedding_dim),
            nn.GELU(),
            nn.Linear(2*embedding_dim, 1)
        )

        self.error_transform = nn.Sequential(
            nn.Linear(embedding_dim, 2*embedding_dim),
            nn.GELU(),
            nn.Linear(2*embedding_dim, embedding_dim),
        )
        self.logical_pred = nn.Sequential(
            nn.Linear(embedding_dim, 2*embedding_dim),
            nn.GELU(),
            nn.Linear(2*embedding_dim, 1),
        )

    
    def forward(self, det_value, p, edge_index, logical_idxs, position_ids, time_ids):
        bs, n_dets = det_value.shape
        det_value = det_value.flatten()
        position_ids = position_ids.flatten()
        time_ids = time_ids.flatten()
        
        p = torch.log(p / 5e-3)
        
        # Combine detector embeddings: value + position + time
        det_value_emb = self.det_value_emb(det_value)
        position_emb = self.position_emb(position_ids)
        time_emb = self.time_emb(time_ids)
        det_value_emb = det_value_emb + position_emb + time_emb
        
        err_value_emb = self.p_embedder(p).squeeze()
        reversed_edge_index = torch.flip(edge_index, dims=[0])
        for i in range(self.num_layers):
            err_value_emb = self.det_to_error[i]((det_value_emb, err_value_emb), edge_index)
            # err_value_emb = self.error_norms[i](err_value_emb)
            det_value_emb = self.error_to_det[i]((err_value_emb, det_value_emb), reversed_edge_index)
            # det_value_emb = self.det_norms[i](det_value_emb)

        err_value_emb = self.final_det_to_err((det_value_emb, err_value_emb), edge_index)

        errs =  self.classifier(err_value_emb)
        relevant_errs = err_value_emb.reshape(bs, -1, self.embedding_dim)[:, logical_idxs] # (bs, n_logical, embedding_dim)
        return errs, self.logical_pred(self.error_transform(relevant_errs).sum(dim=1) / math.sqrt(relevant_errs.shape[1]))
