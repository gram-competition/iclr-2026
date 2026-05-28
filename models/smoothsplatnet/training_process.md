# SmoothSplatNet — Training Process

## Final Submission Form

The final submission entrypoint is `Model()` in `models/smoothsplatnet/model.py`.
It can be instantiated without arguments, loads all weights during construction,
and performs an equal-weight ensemble over three checkpoints.


## Backbone Summary

Each ensemble member uses the same standalone backbone implemented in
`backbone.py`. The model combines:

- [1] Fourier-style coordinate and velocity encodings to improve
  representation of higher-frequency spatial variation.
- [2] A learned monotonic per-axis coordinate warp to adapt the fixed voxel
  grid more smoothly to the geometry distribution.
- [3] Trilinear splatting of pointwise features onto a 3D voxel grid to
  provide structured spatial aggregation while remaining compatible with
  point-cloud inputs.
- [4] A 3D U-Net-style encoder/decoder with skip connections to capture
  multiscale spatial context and recover local detail.
- [5] Squeeze-and-Excitation channel recalibration in the voxel blocks to
  adaptively reweight informative feature channels.
- [6] Residual MLP blocks before and after the voxel network to make
  optimization more stable.
- [7] A small boundary-aware refinement head before the final projection to
  sharpen predictions near the airfoil surface.

## Training Recipe

- Optimizer: `AdamW`[8]
- Peak learning rate: `3e-4`
- Weight decay: `1e-5`
- Learning-rate schedule: `3` warmup epochs followed by cosine decay
- Schedule reference: [9]
- Maximum training length: `150` epochs
- Early stopping patience: `20`
- Validation split: geometry-based holdout, `10%`
- Loss: mean per-point L2 norm over the predicted velocity field

## References

- [1] Tancik et al., 2020, "Fourier Features Let Networks Learn High Frequency
  Functions in Low Dimensional Domains", https://arxiv.org/abs/2006.10739
- [2] Durkan et al., 2019, "Neural Spline Flows",
  https://arxiv.org/abs/1906.04032
- [3] Liu et al., 2019, "Point-Voxel CNN for Efficient 3D Deep Learning",
  https://papers.nips.cc/paper/8382-point-voxel-cnn-for-efficient-3d-deep-learning
- [4] Ronneberger et al., 2015, "U-Net: Convolutional Networks for Biomedical
  Image Segmentation", https://arxiv.org/abs/1505.04597
- [5] Hu et al., 2018, "Squeeze-and-Excitation Networks",
  https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html
- [6] He et al., 2016, "Deep Residual Learning for Image Recognition",
  https://arxiv.org/abs/1512.03385
- [7] Qin et al., 2019, "BASNet: Boundary-Aware Salient Object Detection",
  https://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html
- [8] Loshchilov and Hutter, 2017, "Decoupled Weight Decay Regularization",
  https://arxiv.org/abs/1711.05101
- [9] Loshchilov and Hutter, 2017, "SGDR: Stochastic Gradient Descent with Warm
  Restarts", https://arxiv.org/abs/1608.03983
