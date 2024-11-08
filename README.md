# Embedding Preference Training

Can we train a model that is able to detect good or bad content quality?

## Set-Up

Create a new virtual environment and install all required packages:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## To Run

### Evaluating Embedding Models

For evaluation run the `rerank.py` script from the top directory:

```bash
python3 code/eval/rerank.py
```