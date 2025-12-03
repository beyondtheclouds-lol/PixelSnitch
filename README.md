# PixelSnitch
## General
PixelSnitch is used to detect AI generated images.
## Installation
Copy these commands exactly to install the project

**Clone The Repo**

```git clone https://github.com/beyondtheclouds-lol/PixelSnitch && cd PixelSnitch```

**Make/Activate Virtual Environment**

```python3 -m venv venv && source venv/bin/activate```

**Install Dependencies**

```pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu```

**Unzip Models**

```unzip saved_models/best_model.zip -d saved_models```

## Usage
Make sure this is from inside of your venv

```python app.py```
Navigate to http://localhost:5000

Upload your pictures and click analyze
