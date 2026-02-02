import difflib
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "models/distilbert_imdb_binary"
DATA_DIR = "data/imdb_binary"
SPLITS = ["train", "val", "test"]

# sliding window settings for long reviews
MAX_LENGTH = 256
STRIDE = 128  # how much overlap between chunks
BATCH_SIZE = 16

def load_all_splits():
    """Load train/val/test CSVs and combine them into one dataframe."""
    frames = []
    for split in SPLITS:
        path = f"{DATA_DIR}/{split}.csv"
        df = pd.read_csv(path)

        required_columns = {"movie", "text", "label"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"{path} must contain columns: {required_columns}. Found: {list(df.columns)}"
            )

        df["split"] = split
        df["movie"] = df["movie"].astype(str)
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def pick_movie_title(query, all_titles, cutoff=0.6):
    """
    Try to match the user's query to a movie title.
    First checks for an exact match (case-insensitive), then falls back to fuzzy.
    Returns (title, match_type) or (None, None) if nothing is close enough.
    """
    # build a lowercase -> original mapping for exact matching
    lowercase_to_original = {title.lower(): title for title in all_titles}

    if query.lower() in lowercase_to_original:
        return lowercase_to_original[query.lower()], "exact"

    # nothing exact, try fuzzy
    close_matches = difflib.get_close_matches(query, all_titles, n=1, cutoff=cutoff)
    if not close_matches:
        return None, None

    return close_matches[0], "fuzzy"


def chunk_text(tokenizer, text):
    """
    Split a single review into overlapping chunks using a sliding window.
    This way we don't just silently drop the end of long reviews.
    """
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_tensors="pt",
        padding="max_length",
    )

    # pull each chunk out into its own dict so we can batch them later
    chunks = []
    for i in range(encoded["input_ids"].shape[0]):
        chunks.append({
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
        })

    return chunks


def predict_positive_probs(tokenizer, model, device, reviews):
    """
    Run inference on a list of review texts and return P(positive) for each.
    Long reviews are split into chunks and their scores are averaged.
    """
    model.eval()

    all_positive_probs = []

    with torch.no_grad():
        for review in reviews:
            chunks = chunk_text(tokenizer, review)

            # edge case: empty or unparseable text
            if len(chunks) == 0:
                all_positive_probs.append(0.5)
                continue

            # run chunks through the model in batches
            chunk_scores = []
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]

                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [chunk["input_ids"] for chunk in batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                )
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    [chunk["attention_mask"] for chunk in batch],
                    batch_first=True,
                    padding_value=0,
                )

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

                # grab the positive class column
                chunk_scores.extend(probabilities[:, 1].tolist())

            # average across all chunks for this review
            all_positive_probs.append(float(np.mean(chunk_scores)))

    return np.array(all_positive_probs)


def make_snippet(text, max_chars=240):
    """collapse whitespace, truncate for display"""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "…"


def print_review_block(title, indices, positive_probs, predictions, true_labels, texts):
    """Print a labeled group of reviews (eg: most positive, most ambiguous)"""
    print(f"--- {title} ---")
    for i in indices:
        predicted = "positive" if predictions[i] == 1 else "negative"
        actual = "positive" if true_labels[i] == 1 else "negative"
        print(f"  P(pos): {positive_probs[i]:.3f}  |  predicted: {predicted}  |  label: {actual}")
        print(f"  {make_snippet(texts[i])}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate a sentiment report for a movie.")
    parser.add_argument("--movie", required=True, help="Movie title (exact or approximate)")
    parser.add_argument("--n", type=int, default=300, help="Max reviews to analyze (sampled if more exist)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    device = "mps"
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    df = load_all_splits()
    all_titles = sorted(df["movie"].unique().tolist())

    chosen_title, match_type = pick_movie_title(args.movie, all_titles)
    if chosen_title is None:
        print("No close match found for that movie title, try a different spelling")
        return

    if match_type == "fuzzy" and chosen_title.lower() != args.movie.lower():
        print(f'Using closest match: "{chosen_title}"')
        print()

    movie_df = df[df["movie"] == chosen_title].copy()
    total_available = len(movie_df)
    if total_available == 0:
        print("No reviews found for that title")
        return

    # sample down if there are too many, just for speed
    if total_available > args.n:
        movie_df = movie_df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    reviews = movie_df["text"].tolist()
    true_labels = movie_df["label"].to_numpy()

    # run the model
    positive_probs = predict_positive_probs(tokenizer, model, device, reviews)
    predictions = (positive_probs >= 0.5).astype(int)

    # compute summary stats
    predicted_positive_rate = float((predictions == 1).mean())
    predicted_negative_rate = 1.0 - predicted_positive_rate
    average_prob = float(positive_probs.mean())
    median_prob = float(np.median(positive_probs))

    # how the dataset labels compare (these come from rating buckets, not the model)
    label_positive_rate = float((true_labels == 1).mean())

    # find the most interesting reviews to show
    sorted_by_score = np.argsort(positive_probs)  # low to high
    most_negative = sorted_by_score[:5]
    most_positive = sorted_by_score[-5:][::-1]
    most_ambiguous = np.argsort(np.abs(positive_probs - 0.5))[:5]

    # figure out how many examples to show per category
    # scale it down so we're not showing the same reviews multiple times
    n_reviews = len(movie_df)
    examples_per_group = max(1, min(5, n_reviews // 3))

    print("=== Movie Sentiment Report ===")
    print(f'Title: "{chosen_title}"')
    print(f"Reviews analyzed: {n_reviews:,} (out of {total_available:,} available)")
    print()
    print("Model predictions:")
    print(f"  Positive: {predicted_positive_rate * 100:.1f}%")
    print(f"  Negative: {predicted_negative_rate * 100:.1f}%")
    print(f"  Average P(pos): {average_prob:.3f}")
    print(f"  Median P(pos):  {median_prob:.3f}")
    print()
    print("Dataset labels (on this sample):")
    print(f"  Positive rate: {label_positive_rate * 100:.1f}%")
    print()

    sorted_by_score = np.argsort(positive_probs)  # low to high

    # if there are only a handful of reviews, just show them all in order
    # no point splitting into categories when they'd all overlap
    if n_reviews <= 10:
        print("--- All reviews (sorted by score) ---")
        for i in sorted_by_score[::-1]:  # high to low
            predicted = "positive" if predictions[i] == 1 else "negative"
            actual = "positive" if true_labels[i] == 1 else "negative"
            print(f"  P(pos): {positive_probs[i]:.3f}  |  predicted: {predicted}  |  label: {actual}")
            print(f"  {make_snippet(reviews[i])}")
            print()
    else:
        most_negative = sorted_by_score[:examples_per_group]
        most_positive = sorted_by_score[-examples_per_group:][::-1]
        most_ambiguous = np.argsort(np.abs(positive_probs - 0.5))[:examples_per_group]

        print_review_block("Most positive", most_positive, positive_probs, predictions, true_labels, reviews)
        print_review_block("Most negative", most_negative, positive_probs, predictions, true_labels, reviews)
        print_review_block("Most ambiguous", most_ambiguous, positive_probs, predictions, true_labels, reviews)


if __name__ == "__main__":
    main()
