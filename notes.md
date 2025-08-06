## Logical Observable Placement Effects

**Edge vs Center Performance**: Center-placed logical observables significantly outperform edge placement in CNN decoder.

**Root causes**:
- **Boundary effects**: Edge logicals have fewer neighboring syndrome measurements, creating asymmetric receptive fields for conv layers
- **Error correlation patterns**: Center placement provides richer spatial context - errors can propagate from all directions vs limited patterns at edges  
- **Receptive field utilization**: CNN's ~6-layer receptive field fully utilized for center logicals, partially wasted on padding for edge logicals
- **Training diversity**: Center logicals see more diverse local error environments during training

Problems with a recurrent+conv architecture: the conv per time step needs to be deep, to allow for the possibility of a space-like error. But if it's deep, imagine the computational graph for syndromes that appear early: they go through a huge amount of convs, while syndromes from later go through few. This leads to gradient problems.

Resolution: need to treat time on same footing as space. Conv3d + LSTM or conv3d+attention. key: purely recurrent modeling of time correlations probably inadequate.

If we think about the spacetiem volume of the syndromes, the logical observable lives along a 2d plane right in the middle spatially. This is because flips at all times accumulate, so the final flip value is just the sum of the number of flips that occured along the middle at all times. 

Larger batch size generally improves convergence rate (fitting on a log-log plot, we get bigger slope). This means it is virtually always better for training, because we get a linear slowdown (shifts curve up and down in terms of training time) but polynomial speedup in convergence (increased slope).

General phenomenon is that deep circuits train slower (they have a smaller slope on a log-log plot - this is not just a constant offset, this means they actually converge much slower), like 1/t^0.9 instead of 1/t^2.5 (these are the rates we can achieve for super shallow circuits). Two options:
- Make the model wider (i.e., increase embedding dimension). This improves things, but not appreciably.
- Make the model deeper (this improves things -- e.g., resnet50 vs 101). This appears to be the solution

doubling embedding dimension vs doubling depth: depth wins -- this is nice anyways, because the training memory cost is linear in depth and quadratic in embedding dimension

revision: increasing model power (e.g., depth or width) only results in wins *if* we can keep the batch size large enough. Experiments on d~13 shows resnet101 beating resnet50, etc in terms of convergence rate, but both have batch sizes that are very large (~thousands). But on d~19, we start to get limited by memory and resnet50 with downsampling in later layers (much lower memory pressure) beats resnet50 with no downsampling and resnet101 -- this is because the batchsize is much larger. So when we scale architecture, we need to ensure the batch size keeps up, otherwise we won't get the wins from scaling architecture.

Takeaway: decoding with larger distance is not really the hard problem. The hard problem is training with deeper circuits. Solution: use deeper networks. This of course makes inference (and training, but who cares about training) slower by a factor proportional to the depth, but maybe distillation can help.
