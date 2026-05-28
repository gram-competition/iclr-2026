# TransolverResidual — GRaM 2026 Submission

This is our submission for the GRaM challenge at ICLR 2026.
Authors: Mikel Mendibe, Ivan Bioli, Massimiliano Ghiotto.

**Task:** Predict 5 future velocity field timesteps from 5 input timesteps on irregular
3D point clouds (~100k points) around F1-style airfoil geometries.

## Approach:
To be honest we were quite wary of computational-constraints, so our approach was mainly to try to circumvent those problems. Taking the time limitations into account, we wanted to leverage an architecture that is not too demanding computationally, and in the end, we settled on the architecture Transolver [1] paper, cited below. Looking back, and having in mind that this workshop was centered around "Geometry-grounded Representation Learning", we believe this was a mistake, but hey, you win some you lose some.

It is obvious that these simulations have a very easy part (the laminar flow) and a very difficult part (the turbulent flow). So, in order to use this inductive bias somehow, we framed the model in the following way:
First, using the first 5 snapshots, we build a polynomial of degree 2 approximation for each point, which acts as our baseline. Then, we use the Transolver to try and learn the residual. (Was this sensible? I don't really know.)

The Transolver architecture divides the points from the point-cloud in different slices, in theory depending on the "regime" in which they are. This is, points that are on the wake, should be on the same slice, points that are on the clean laminar flow should be on the same slice... Then, Physics Attention is applied among these slices, which gives the model some sort of global knowledge. Howeeeeever, as you can imagine, points don't get to know anything about their local neighborhood, which is NOT good. We tried to "bandaid" this by running some kNN lookup at the beginning and computing the mean velocities of the 8 closest points to each point, as a way of adding some local information, but this is very poor and quite a senseless idea.

Another problem we face the ability to include the temporal inductive bias (spoiler, we don't have any, at least for the residual part). The polynomial part respects time in a proper way, but we couldn't find any "cheap" way of adding a sensible way of incorporating the temporal inductive bias within the Transolver (maybe we could try to frame it as a Markovian solve where we always predict a timestep starting from the previous ones, but we didn't do it). So, of course, we tried to "bandaid" it again, in this case by adding more feature engineering and adding some temporal_deltas as an input vector. Does it really make it any better? Not sure, I would be surprised it makes too big of a difference, it just adds more features.


And last but not least, I wouldn't say our solution is too "Geometry Grounded". it's true that the physics regimes that we can find in this simulation are quite geometry agnostic, but still, we only "encode" geometry by adding to the feature vector of each point: (i) distance to the nearest surface point, (ii) signed x-offset to the nearest surface point. This is not nearly enough, we should have tried to encode this geometry more faithfully and in a more "global" manner (some message passing?).



Anyways, the results are not too bad (we hope), and some decent tricks were applied (the polynomial baseline, some data augmentation...). If you inspect the soft-assignment of points to the slices, you get mixed feelings, it's true that for some layers the entropy is low, which means that points were clearly divided, which is good :-)
Some of these slices end up empty, so we know that our model has too many slices, maybe it would've been wise to analyze this in a proper way instead of putting 32 slices just for the sake of it. 
In other slices, the entropy is super-high and we see that points were randomly and evenly distributed among slices (not good), but as we have skip connections, this is not that big of a problem.

All in all:
- Cool experience B-)
- I hope that the people who read this can learn from our mistakes. If you find some clever ways to circumvent our problems or want to shed some light on us, it would be more than welcome.

Big thank you to the sponsors and organizers!!!!!


---

## Training

**Data**

- 905 samples (181 simulations × 5 time windows), all used for training.
- 90 / 10 random train / val split.
- Each sample: `velocity_in (5, 100k, 3)`, `pos (100k, 3)`, `idcs_airfoil`, `velocity_out (5, 100k, 3)`.

**Preprocessing**

Distance features (`dist_to_airfoil`, `upstream_dist`, `is_airfoil`) and k-NN indices
are precomputed per simulation and stored as `.distcache.npz` / `.knncache.npz` sidecar
files to avoid recomputing them every epoch.

**Loss**

Relative L² loss on the velocity (predicted output vs. ground truth):

```
loss = ‖ velocity_out_pred − velocity_out_gt ‖₂ / ‖ velocity_out_gt ‖₂
```

**Optimiser**

Adam (lr=1e-3) with cosine annealing over 400 epochs.  
Gradient accumulation over 4 steps (effective batch size 4 with batch_size=1).


**Hardware**

Single NVIDIA L40S (40 GB).  Training time: ~6 hours for 400 epochs.


**Reproduction**

```
bash runner.sh
```

---

## Inference Notes

The model caches per-geometry distance features and k-NN indices by geometry fingerprint.
The first call on a novel geometry (not seen during training) incurs a one-time precompute
cost (~5–10 seconds for 100k points on CPU).  Subsequent calls on the same geometry
(e.g. different time windows of the same simulation) are instant.

The model is fully self-contained: no external config files are required.
`TransolverResidual()` instantiates with trained weights loaded automatically.

## Disclaimer
A good part of the code was either written or re-written with the assitance of LLMs. Not really happy about it, but the deadline was tight, and hey, we had to make some sacrifices.

---

## References

[1] Wu, H., Luo, H., Wang, H., Wang, J. &amp; Long, M.. (2024). Transolver: A Fast Transformer Solver for PDEs on General Geometries. <i>Proceedings of the 41st International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 235:53681-53705 Available from https://proceedings.mlr.press/v235/wu24r.html.

