# Sliding-Window Logical-Flip Decoding — Reference Note

## 1 Why sliding windows?
* **Bounded latency & memory** – decode only a finite slice of *w* syndrome rounds instead of the full 3-D spacetime graph.  
* **Trade-off** – larger *w* → lower logical-error rate but higher latency (≈ *w* cycles) and RAM.

---

## 2 Adapting a “logical-only” decoder
A global end-of-run logical-flip predictor doesn’t compose across windows.  
You need **incremental, window-consistent outputs.**

| Approach | Change needed | Cycle-latency | Typical use |
|----------|---------------|---------------|-------------|
| **Window-cropped feed-forward** | Re-train on fixed-width windows; predict if a flip **starts** inside the commit region. | *w* | Moderate-latency controllers |
| **Recurrent / GRU head** | Add state; emit a flip bit every round. | 1–2 | Ultra-low-latency or feedback-critical circuits |

---

## 3 Window / commit scheme (common choice)
* **Window length** *w*  
* **Stride** *s = w⁄2* (50 % overlap)  
* **Commit region** = first *w⁄2* rounds; remaining *w⁄2* rounds are look-ahead context.

**Example – w = 40**

window rounds : 0 … 39
commit rounds : 0 … 19 (actions & labels)
context : 20 … 39 (future look-ahead)
next window : 20 … 59


---

## 4 Label definition (“flip inside commit region”)
1. Keep incoming Pauli-frame bit `F_in`.  
2. For rounds in the **commit region only**, set target = 1 if the logical parity toggles relative to `F_in`; else 0.  
3. Errors that began earlier or finish later are handled by adjacent windows, ensuring a **globally consistent frame history.**

---

## 5 Minimal training recipe (toy demo)

| Parameter | Value |
|-----------|-------|
| Code | Surface, distance *d = 5* |
| Window | *w = 2d = 10*, stride = 10 (no overlap) |
| Target | Binary: Z-flip starts in rounds 0-9 |
| Model | Tiny 3-D CNN (~20 k params) |
| Loss | BCE with `pos_weight ≃ 1/p_flip` |
| Data | Millions of simulated windows (STIM + phenomenological noise) |

Python-like pseudocode (indented four spaces to render as code):

    class TinyWindowNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
                nn.Conv3d(8,16,3, padding=1),  nn.ReLU(),
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                nn.Linear(16,1)
            )
        def forward(self, x):          # x shape (B,1,w,Lx,Ly)
            return self.net(x).squeeze(1)   # logits

Train until validation AUROC saturates (≈ 0.99 typical for *d = 5*).

---

## 6 Maintaining exponential error suppression
* **Overlap windows** (*stride = w⁄2*) *or* **defer a short tail** (commit only first *w − m* rounds).  
* This guarantees every measurement error appears at least once as an **interior** defect pair → retains distance-*d* scaling:

\[
p_L \;\approx\; \bigl(c\,p_{\text{phys}}\bigr)^{\lfloor d_{\text{eff}}/2\rfloor},
\qquad d_{\text{eff}} = \min\!\bigl(d,\;w⁄2\bigr).
\]

---

## 7 Deployment loop (stateless variant)

Indented pseudocode:

    buffer = deque(maxlen=w)          # stores last w syndrome frames
    for each new round:
        buffer.append(read_syndrome())
        if len(buffer) == w:          # buffer full
            logits = model(to_tensor(buffer))
            if torch.sigmoid(logits) > τ:
                logical_Z ^= 1        # toggle Pauli frame
            for _ in range(s):        # drop s = w//2 frames
                buffer.popleft()

Latency = *s* cycles (e.g. 20 cycles for *w = 40, s = 20*).

---

## 8 When to switch to the recurrent variant
* Need < 10-cycle latency  
* Adaptive mid-circuit feed-forward (e.g., lattice-surgery teleportation)  
* Streaming hardware that processes one round at a time  

Add a small GRU/LSTM head, relabel per round, and train with class-imbalance-aware loss.
