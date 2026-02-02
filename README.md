# IMDb Review Sentiment Analysis with DistilBERT

## Overview

This project fine-tunes a pretrained DistilBERT transformer to classify IMDb movie reviews as positive vs negative. It includes:

- Preprocessing from a Kaggle IMDb reviews dataset (rating + review text + movie title)
- Leakage-aware data splitting (grouped by movie)
- Transformer fine-tuning using Hugging Face Trainer
- Evaluation + error analysis (confusion matrix, confidence histograms)
- A CLI tool that generates a sentiment report for any movie in the dataset

## Dataset & Labels

**Source:** Kaggle IMDb reviews dataset (50k+ rows)

**Binary label mapping:**
- Ratings 1–4 → negative (0)
- Ratings 7–10 → positive (1)
- Ratings 5–6 are dropped to avoid ambiguous sentiment

Each row contains: `movie`, `text`, `rating`, `label`

## Preventing Movie Leakage

Reviews are split by movie title using group-based splitting so that all reviews for the same movie appear in only one of train/val/test. This prevents the model from "cheating" by learning movie-specific keywords that would otherwise appear across splits.

## Model

- **Base model:** `distilbert-base-uncased`
- **Task head:** Sequence classification (num_labels=2)
- **Tokenization:** Subword tokenization + attention masks
- **Max length:** 256 tokens
- **Long-review handling:** In the reporting tool, sliding-window chunk inference is used to reduce truncation errors

## Results

- **Accuracy:** ~0.92 on held-out test set
- **F1:** ~0.92 on held-out test set

(Exact numbers may vary due to sampling and training settings.)

## How to Run

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets scikit-learn pandas matplotlib numpy evaluate
```

### 2. Preprocess

Creates `data/imdb_binary/{train,val,test}.csv`:

```bash
python3 src/preprocess.py
```

### 3. Train

Fine-tunes DistilBERT and saves the model to `models/distilbert_imdb_binary`:

```bash
python3 src/train.py
```

### 4. Evaluate + Visualize

Prints metrics and shows plots (confusion matrix + confidence histograms):

```bash
python3 src/test_visualize.py
```

### 5. Generate a Movie Sentiment Report

Aggregates model predictions across reviews for a movie and prints examples:

```bash
python3 src/movie_report.py --movie "Fight Club" --n 200
```

### 6. Interactive Testing (Optional)

Test individual reviews interactively:

```bash
python3 src/test_model.py
```

## Error Analysis Notes

Common failure modes found:

- **Truncation:** Long reviews can flip sentiment if important cues appear after the first N tokens → mitigated in the report tool using chunked inference
- **Label noise:** Rating-based labels sometimes conflict with the review text (e.g., sarcastic or inconsistent reviews)
