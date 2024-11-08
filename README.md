# Embedding Preference Training

Can we train a model that is able to detect good or bad content quality?

## Set-Up

Create a new virtual environment and install all required packages:

```bash
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

### Scraping Data

This code is still being generalized, but in general, scripts to extract and save data will be found in `dataset/scraping/scrape_*.py`:

```bash
python3 code/dataset/scraping/scrape_gb_wiki.py
```

### Train Binary Classifier

To experiment with training binary classifiers, run the `train_classifier.py` script from the top directory:

```bash
python3 code/models/train_classifier.py
```
