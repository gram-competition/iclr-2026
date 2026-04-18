# FNO_DSE_TIME Competition Model

This directory contains the FNO3d_dse_v2 model packaged for the ICLR 2026 GRaM competition.

- Model class: `FNO_DSE_TIME` (imported from `models.fno_dse_time`)
- Weights: Loaded automatically from `output/20260413-005251_fno_dse_time/checkpoints/best.pt` (relative to repo root)
- Usage:

```python
from models import FNO_DSE_TIME
model = FNO_DSE_TIME()
output = model(t, pos, idcs_airfoil, velocity_in)
```

- Model source: See `model.py` for the full implementation.
- The model is self-contained and does not require arguments to instantiate.
