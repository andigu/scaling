import stim
import numpy as np
from tqdm.auto import tqdm, trange

circuit_gen = lambda p, distance, rounds: stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=rounds,
    distance=distance,
    after_clifford_depolarization=p,
    after_reset_flip_probability=p,
    before_measure_flip_probability=p,
    before_round_data_depolarization=p).flattened()

circuit = circuit_gen(0.0075, 3, 3)

z_detectors = []
z_det_only = stim.Circuit()
for line in circuit:
    if line.name == 'DETECTOR':
        pos_x, pos_y, time = map(int, line.gate_args_copy())
        if time == 0:
            z_detectors.append((pos_x, pos_y))
            z_det_only.append(line)
        else:
            if (pos_x, pos_y) in z_detectors:
                z_det_only.append(line)
    else:
        z_det_only.append(line)

dem = z_det_only.detector_error_model(decompose_errors=False)
error_count = 0
detector_coords = []
error_probs = []
dem_graph = []
errors_for_logical = []
for line in dem:
    if line.type == 'error':
        p = line.args_copy()
        error_probs.append(p)
        targets = line.target_groups()
        if len(targets) == 1:
            for target in targets[0]:
                if target.is_logical_observable_id():
                    errors_for_logical.append(error_count)
                elif target.is_relative_detector_id():
                    dem_graph.append((target.val, error_count))
            error_count += 1
        else:
            raise ValueError("Not sure how to handle this.")
    elif line.type == 'detector':
        detector_coords.append(list(map(int, line.args_copy())))

def batch_graph(edges, batch_size: int):
    n_nodes_left = edges[0].max() + 1
    n_nodes_right = edges[1].max() + 1
    offset = (np.arange(batch_size)[:, None] * np.array([n_nodes_left, n_nodes_right]))[..., None]
    edges = edges[None, ...]
    edges = (edges + offset).transpose((1,0,2))
    
    edges = edges.reshape(2, -1)
    return edges

dem_graph = np.array(dem_graph).T
errors_for_logical = np.array(errors_for_logical) # Which error ids flip the logical observable
detector_coords = np.array(detector_coords)
error_probs = np.array(error_probs).flatten()
positions = detector_coords[:,:2]
position_id = np.unique(positions, axis=0, return_inverse=True)[1]
time_id = detector_coords[:,2]

# Create position_id mapping
unique_positions = sorted(list(set([(x, y) for x, y, t in detector_coords])))
position_to_id = {pos: i for i, pos in enumerate(unique_positions)}

# Create position_ids and time_ids arrays
position_ids = np.array([position_to_id[(x, y)] for x, y, t in detector_coords])
time_ids = detector_coords[:, 2]  # Time is the third coordinate


from model import Decoder
import torch
torch.set_float32_matmul_precision('medium')
import matplotlib.pyplot as plt
import pymatching
import schedulefree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dec = torch.compile(Decoder(embedding_dim=256), mode='max-autotune').to(device)
sampler = dem.compile_sampler(seed=0)

# Create MWPM decoder
mwpm = pymatching.Matching.from_detector_error_model(dem)

bs = 2048
optimizer = schedulefree.RAdamScheduleFree(dec.parameters(), lr=1e-3)

# Create MLP baseline model
import torch.nn as nn
X_sample, _, _ = sampler.sample(shots=1, return_errors=True)
mlp = nn.Sequential(
    nn.Linear(X_sample.shape[1], 1024),
    nn.GELU(),
    nn.Linear(1024, 512),
    nn.GELU(),
    nn.Linear(512, 1)
).to(device)
mlp_optimizer = schedulefree.RAdamScheduleFree(mlp.parameters(), lr=1e-3)

# Test MWPM baseline performance
print("Evaluating MWPM baseline...")
test_shots = 100000
X_test, obs_test, _ = sampler.sample(shots=test_shots, return_errors=True)
mwpm_pred = mwpm.decode_batch(X_test)
mwpm_accuracy = (mwpm_pred == obs_test).mean()
mwpm_error_rate = 1 - mwpm_accuracy
print(f"MWPM baseline error rate: {mwpm_error_rate:.4f} (accuracy: {mwpm_accuracy:.4f})")

# Initialize EMA tracking
gnn_error_rate_ema = None
mlp_error_rate_ema = None
raw_error_loss_ema = None

# History tracking  
gnn_error_rate_history = []
mlp_error_rate_history = []
raw_error_loss_history = []


batched_graph = batch_graph(dem_graph, bs)
batched_p = np.tile(error_probs, bs)[:, None]
batched_p = torch.from_numpy(batched_p).float().to(device)
batched_graph = torch.from_numpy(batched_graph).to(device)
errors_for_logical = torch.from_numpy(errors_for_logical).to(device)
# Batch position_ids and time_ids
batched_position_ids = np.tile(position_ids, bs)
batched_time_ids = np.tile(time_ids, bs)
batched_position_ids = torch.from_numpy(batched_position_ids).int().to(device)
batched_time_ids = torch.from_numpy(batched_time_ids).int().to(device)

optimizer.train()
mlp_optimizer.train()
with trange(2500) as t:
    for i in t:
        # Set models to train mode for schedulefree
        X, obs, raw_err = sampler.sample(shots=bs, return_errors=True)
        X = torch.from_numpy(X).int().to(device)
        obs = torch.from_numpy(obs).float().to(device).flatten()
        raw_err = torch.from_numpy(raw_err).float().to(device).flatten()
        optimizer.zero_grad()
        
        
        raw_err_pred, logical_pred = dec(X, batched_p, batched_graph, errors_for_logical, 
                                         batched_position_ids, batched_time_ids)
        err_loss = torch.nn.functional.binary_cross_entropy_with_logits(raw_err_pred.flatten(), raw_err)
        logical_loss = torch.nn.functional.binary_cross_entropy_with_logits(logical_pred.flatten(), obs)
        logical_loss_weight = min(i/2000, 1.0)
        loss = logical_loss_weight * logical_loss + (1-logical_loss_weight) * err_loss
        loss.backward()
        optimizer.step()
        
        # Train MLP baseline
        mlp_optimizer.zero_grad()
        mlp_pred = mlp(X.float())
        mlp_loss = torch.nn.functional.binary_cross_entropy_with_logits(mlp_pred.flatten(), obs)
        mlp_loss.backward()
        mlp_optimizer.step()
        
        # Calculate error rates (1 - accuracy)
        with torch.no_grad():
            # Set to eval mode for inference
            
            gnn_pred_binary = (logical_pred > 0).float()
            gnn_accuracy = (gnn_pred_binary.flatten() == obs).float().mean().item()
            gnn_error_rate = 1 - gnn_accuracy
            
            mlp_pred_binary = (mlp_pred > 0).float()
            mlp_accuracy = (mlp_pred_binary.flatten() == obs).float().mean().item()
            mlp_error_rate = 1 - mlp_accuracy
        
        # Update EMAs
        alpha = 0.01
        if gnn_error_rate_ema is None:
            gnn_error_rate_ema = gnn_error_rate
            mlp_error_rate_ema = mlp_error_rate
            raw_error_loss_ema = err_loss.item()
        else:
            gnn_error_rate_ema = (1 - alpha) * gnn_error_rate_ema + alpha * gnn_error_rate
            mlp_error_rate_ema = (1 - alpha) * mlp_error_rate_ema + alpha * mlp_error_rate
            raw_error_loss_ema = (1 - alpha) * raw_error_loss_ema + alpha * err_loss.item()
        
        # Store history
        gnn_error_rate_history.append(gnn_error_rate_ema)
        mlp_error_rate_history.append(mlp_error_rate_ema)
        raw_error_loss_history.append(raw_error_loss_ema)
        
        t.set_postfix(gnn_err=f"{gnn_error_rate_ema:.4f}", mlp_err=f"{mlp_error_rate_ema:.4f}", raw_err_loss=f"{raw_error_loss_ema:.4f}")

# Plot error rate history
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
step = np.arange(len(gnn_error_rate_history)) + 1

# Plot logical error rates
ax1.loglog(step, gnn_error_rate_history, label='GNN', linewidth=2)
ax1.semilogy(step, mlp_error_rate_history, label='MLP', linewidth=2)
# ax1.axhline(y=mwpm_error_rate, color='red', linestyle='--', label=f'MWPM baseline: {mwpm_error_rate:.4f}', linewidth=2)

ax1.set_xlabel('Training Step')
ax1.set_ylabel('Error Rate (1 - Accuracy)')
ax1.set_title('Logical Error Rate Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
# ax1.set_ylim(bottom=mwpm_error_rate/2)

# Plot raw error loss
ax2.loglog(step, raw_error_loss_history, label='Raw Error Loss', linewidth=2, color='green')
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Binary Cross-Entropy Loss')
ax2.set_title('Raw Error Prediction Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_rate_history.png', dpi=150)

print(f"\nFinal error rates:")
print(f"  MWPM baseline: {mwpm_error_rate:.4f}")
print(f"  GNN: {gnn_error_rate_ema:.4f}")
print(f"  MLP: {mlp_error_rate_ema:.4f}")
print(f"  Raw error loss: {raw_error_loss_ema:.4f}")
print("Error rate plot saved to 'error_rate_history.png'")