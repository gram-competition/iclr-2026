# Wavelet Latent Operator
## Author
Alex Brown  
https://scholar.google.co.uk/citations?user=PF1hkt8AAAAJ&hl=en  
https://www.linkedin.com/in/alex-brown-bb1110230/

## Overview
A geometry-conditioned latent operator that encodes 3D airflow history via solid harmonic wavelet scattering, processes them in a 3D convolutional latent space modulated by a geometry context vector, and decodes future velocities at each point as a linear combination of local history basis vectors.

## Architecture
**Geometry encoding.** Airfoil surface points are voxelised onto a 48³ occupancy grid via trilinear splatting. Second-order solid harmonic wavelet scattering (J=3, L=3, building off Kymatio) is applied to the occupancy volume, producing a 164-dimensional Lp pooled geometry descriptor per sample.

**History encoding.** The 5-step velocity history is decomposed into a mean field and a residual. Subtracting the mean field before scattering means the module encodes flow variation rather than absolute velocity, with the mean recovered at decode time as part of the local basis. The residual (vx, vy, vz, speed) is voxelized per timestep. We extract features using second-order solid harmonic wavelet scattering, but this time preserving spatial responses and pooling to 12³. This gives a (5 × C_maps × 12³) history tensor per sample, where C_maps = 160 (4 channels × 40 scattering maps each), precomputed and cached before training.

**Latent operator.** Geometry influences the operator three ways. First, the 164-dim scattering descriptor produces 160 sigmoid gates -- one per history map channel -- which weight the scattering channels by their relevance to the wing geometry: given this wing shape, which velocity features matter. Second, the 48³ occupancy volume is average-pooled to 12³ and concatenated with the gated history maps and a [-1,1]³ coordinate grid, giving the network direct spatial access to wing geometry at each voxel. Third, the scattering descriptor is projected to a 64-dim context vector that modulates every residual block via FiLM [1]. The concatenated input is lifted to 64 channels by a 1×1 Conv3d, then processed by 4 residual blocks (GroupNorm + Conv3d × 2, GELU, tanh-clamped FiLM), with the context vector injected at each block. A final 1×1 head expands to 5 × 64 channels, reshaped to give one 64-channel latent volume per output step.

**Point decoder.** For each output step, per-point latent features are sampled from the latent volume via trilinear interpolation. A 2-layer MLP with LayerNorm decodes 7 scalar coefficients which linearly combine a local 7-vector basis: the 5 history velocity vectors, their mean, and the last temporal delta. Coefficients are bounded via tanh × 0.35, constraining each output step to a small correction around the local basis rather than an unconstrained prediction. This basis formulation constrains the model to predict physically plausible corrections -- future velocity is expressed as a weighted combination of observed flow states rather than an unconstrained extrapolation. The final prediction adds the repeat-last baseline which carries much of the low frequency signal and enforces zero velocity on airfoil surface points.

## Training
- **Loss:** MSE + 0.2 × per-sample relative L2 + 0.05 × temporal-difference MSE
- **Optimiser:** AdamW, lr=3e-4, weight decay=1e-5
- **Schedule:** ReduceLROnPlateau (patience=5, factor=0.5, min lr=1e-6)
- **Epochs:** 75, batch size 2, gradient clipping at 1.0. Best checkpoint saved by validation relative L2.
- **Split:** 85/15 stratified by simulation ID
- **Normalisation:** residual velocities normalised to zero mean / unit std computed over a random subsample of training points

Wavelet feature caches (geometry volumes, scattering descriptors, history maps) were precomputed once on GPU before training and loaded from disk each epoch.

## Additional Thoughts
Due to time constraints and compute limitations, little exploration of the hyperparamter space was undertaken. It is reasonable to expect that with a little empirical experimentation, one could improve upon our results substantially. We also acknowledge that voxelisation moves away from the native point cloud structure of the problem -- a mesh-native or point-based operator would be a more principled fit for this data.

## References
[1] Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018. https://arxiv.org/abs/1709.07871
