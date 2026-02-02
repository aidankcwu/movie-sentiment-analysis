import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


MODEL_DIR = "models/distilbert_imdb_binary"
DATA_DIR = "data/imdb_binary"
MAX_LENGTH = 256
BATCH_SIZE = 32
TOP_K_ERRORS = 10  # how many FP/FN examples to print

def load_split(path):
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain columns: text, label. Found: {list(df.columns)}")
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def predict(df, tokenizer, model, device):
    ds = Dataset.from_pandas(df[["text", "label"]])

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds_tok = ds.map(tok, batched=True, remove_columns=["text"])
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    def collate_fn(batch):
        labels = torch.stack([item["label"] for item in batch])
        batch_wo_labels = [{k: v for k, v in item.items() if k != "label"} for item in batch]
        collated = DataCollatorWithPadding(tokenizer=tokenizer)(batch_wo_labels)
        collated["label"] = labels
        return collated

    dl = DataLoader(ds_tok, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_probs, all_preds, all_labels = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in dl:
            labels = batch["label"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items() if k != "label"}

            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

    probs = np.vstack(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return labels, preds, probs


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["NEG", "POS"])
    plt.yticks([0, 1], ["NEG", "POS"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_normalized(cm, title):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm_norm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["NEG", "POS"])
    plt.yticks([0, 1], ["NEG", "POS"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_confidence_hist_correct_incorrect(confidence, correct_mask, title):
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(confidence[correct_mask], bins=30, alpha=0.7, label="Correct")
    plt.hist(confidence[~correct_mask], bins=30, alpha=0.7, label="Incorrect")
    plt.title(title)
    plt.xlabel("Model confidence (max softmax probability)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confidence_hist_fp_fn(confidence, fp_mask, fn_mask, title):
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(confidence[fp_mask], bins=30, alpha=0.7, label="False Positives (NEG→POS)")
    plt.hist(confidence[fn_mask], bins=30, alpha=0.7, label="False Negatives (POS→NEG)")
    plt.title(title)
    plt.xlabel("Model confidence (max softmax probability)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_length_vs_confidence(lengths, confidence, correct_mask, title):
    plt.figure(figsize=(6.5, 4.2))
    plt.scatter(lengths, confidence, alpha=0.35)
    plt.title(title)
    plt.xlabel("Review length (characters)")
    plt.ylabel("Model confidence")
    plt.tight_layout()
    plt.show()

    # optional: compare distributions for correct vs incorrect
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(lengths[correct_mask], bins=40, alpha=0.7, label="Correct")
    plt.hist(lengths[~correct_mask], bins=40, alpha=0.7, label="Incorrect")
    plt.title(title + " (length distribution)")
    plt.xlabel("Review length (characters)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_top_errors(df, labels, preds, probs, k=10):
    prob_pos = probs[:, 1]
    confidence = probs.max(axis=1)

    fp_mask = (labels == 0) & (preds == 1)  # NEG predicted POS
    fn_mask = (labels == 1) & (preds == 0)  # POS predicted NEG

    fp_idx = np.where(fp_mask)[0]
    fn_idx = np.where(fn_mask)[0]

    # sort by confidence descending
    fp_sorted = fp_idx[np.argsort(-confidence[fp_idx])] if len(fp_idx) > 0 else []
    fn_sorted = fn_idx[np.argsort(-confidence[fn_idx])] if len(fn_idx) > 0 else []

    def show_examples(indices, title):
        print(f"\n--- {title} (top {min(k, len(indices))}) ---")
        for i in indices[:k]:
            text = df.iloc[i]["text"]
            text_snip = (text[:300] + "…") if len(text) > 300 else text
            true_y = labels[i]
            pred_y = preds[i]
            conf = confidence[i]
            ppos = prob_pos[i]
            print(f"\nidx={i} | true={true_y} pred={pred_y} | conf={conf:.3f} | P(pos)={ppos:.3f}")
            print(text_snip)

    show_examples(fp_sorted, "False Positives (NEG→POS)")
    show_examples(fn_sorted, "False Negatives (POS→NEG)")


def evaluate_split(name, df, tokenizer, model, device):
    labels, preds, probs = predict(df, tokenizer, model, device)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    confidence = probs.max(axis=1)
    correct = (preds == labels)

    fp_mask = (labels == 0) & (preds == 1)
    fn_mask = (labels == 1) & (preds == 0)

    lengths = df["text"].str.len().to_numpy()

    print(f"\n=== {name.upper()} RESULTS ===")
    print(f"Rows:     {len(df):,}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Visualization
    plot_confusion_matrix(cm, title=f"{name.upper()} Confusion Matrix (Counts)")
    plot_confusion_matrix_normalized(cm, title=f"{name.upper()} Confusion Matrix (Normalized)")
    plot_confidence_hist_correct_incorrect(confidence, correct, title=f"{name.upper()} Confidence: Correct vs Incorrect")
    plot_confidence_hist_fp_fn(confidence, fp_mask, fn_mask, title=f"{name.upper()} Confidence: FP vs FN")
    plot_length_vs_confidence(lengths, confidence, correct_mask=correct, title=f"{name.upper()} Length vs Confidence")

    # Print top errors
    print_top_errors(df, labels, preds, probs, k=TOP_K_ERRORS)


def main():
    device = "mps"
    print(f"Loading model from: {MODEL_DIR}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    val_df = load_split(f"{DATA_DIR}/val.csv")
    test_df = load_split(f"{DATA_DIR}/test.csv")

    evaluate_split("val", val_df, tokenizer, model, device)
    evaluate_split("test", test_df, tokenizer, model, device)


if __name__ == "__main__":
    main()