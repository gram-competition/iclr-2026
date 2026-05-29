# Submission

will be documenting the whole journey in detail later!

# Structure

models/zonal_moe/
├── __init__.py
├── model.py
├── preprocessing.py
├── train.py
├── wrapper.py
├── weights.pt
└── README.md

- `model.py` — model arch file
- `preprocessing.py` — helper for building KNN and other metric computations
- `train.py` — main training entry 
- `wrapper.py` — wraps the MOE model for inference 
- `weights.pt` — pretrained model weights (5.5 MB)
