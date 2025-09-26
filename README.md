# PixelSnitch

# Deepfake Detector (Phase 1: Images)

A minimal, working image deepfake detector for capstone Phase 1.  
Two baselines:
- **EfficientNet-B0 (PyTorch)** transfer learning
- **ELA + LogisticRegression** classical baseline

## Setup

```bash
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
