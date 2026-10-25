import os
import random
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk

import data_loader
import phonetic as phonetic_module
import wsd as wsd_module

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Stopwords for overlap check
_STOPWORDS = {"the", "a", "an", "and", "or", "of", "in", "to", "for",
              "with", "on", "at", "from", "by", "is", "are", "was", "were",
              "be", "been", "being", "have", "has", "had"}


def _label_words(cui_label):
    tokens = word_tokenize(cui_label.lower())
    return {t for t in tokens if t.isalpha() and len(t) >= 3 and t not in _STOPWORDS}


def _definition_words(synset):
    if synset is None:
        return set()
    tokens = word_tokenize(synset.definition().lower())
    return {t for t in tokens if t.isalpha() and len(t) >= 3 and t not in _STOPWORDS}


def sense_matches(predicted_synset, cui_label):
    """Return True if the synset definition shares meaningful words with the UMLS label."""
    if predicted_synset is None:
        return False
    label_w = _label_words(cui_label)
    def_w = _definition_words(predicted_synset)
    return bool(label_w & def_w)


def evaluate_algorithm_on_term(algorithm_fn, term, instances):
    """Run algorithm_fn on all instances for one term and return accuracy stats."""
    correct = 0
    unevaluable = 0

    for inst in instances:
        sentence = inst["sentence"]
        cui_label = inst["cui_label"]

        if not wn.synsets(term):
            unevaluable += 1
            continue

        try:
            predicted = algorithm_fn(sentence, term)
        except Exception:
            predicted = None

        if predicted is None:
            unevaluable += 1
            continue

        if sense_matches(predicted, cui_label):
            correct += 1

    evaluable = len(instances) - unevaluable
    accuracy = correct / evaluable if evaluable > 0 else 0.0
    coverage = evaluable / len(instances) if instances else 0.0

    return {
        "correct": correct,
        "total": len(instances),
        "unevaluable": unevaluable,
        "evaluable": evaluable,
        "coverage": round(coverage, 3),
        "accuracy": round(accuracy, 3),
    }


def run_task2(dataset, n_regular=10, n_acronyms=10, seed=42):
    """Evaluate NLTK Lesk on sampled terms. Returns (results_df, summary_df)."""
    random.seed(seed)
    regular_terms, acronym_terms = data_loader.sample_terms(
        dataset, n_regular=n_regular, n_acronyms=n_acronyms, seed=seed
    )

    selected = (
        [(t, "regular") for t in regular_terms] +
        [(t, "acronym") for t in acronym_terms]
    )

    rows = []

    def nltk_lesk_fn(sentence, word):
        tokens = word_tokenize(sentence.lower())
        return lesk(tokens, word)

    for term, term_type in selected:
        instances = dataset[term]["instances"]
        for inst in instances:
            sentence = inst["sentence"]
            cui = inst["cui"]
            cui_label = inst["cui_label"]

            has_wn = bool(wn.synsets(term))
            if not has_wn:
                predicted_str = "not_in_wordnet"
                correct = None
            else:
                tokens = word_tokenize(sentence.lower())
                try:
                    predicted = lesk(tokens, term)
                except Exception:
                    predicted = None

                predicted_str = predicted.name() if predicted else "no_prediction"
                correct = sense_matches(predicted, cui_label) if predicted else None

            rows.append({
                "term": term,
                "term_type": term_type,
                "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                "ground_truth": cui_label,
                "predicted_sense": predicted_str,
                "correct": correct,
            })

    results_df = pd.DataFrame(rows)
    summary_rows = []
    for ttype in ["regular", "acronym"]:
        subset = results_df[results_df["term_type"] == ttype]
        evaluable = subset[subset["correct"].notna()]
        n_samples = len(evaluable)
        n_correct = int(evaluable["correct"].sum()) if n_samples > 0 else 0
        acc = n_correct / n_samples if n_samples > 0 else 0.0
        summary_rows.append({
            "Category": ttype.capitalize() + " terms",
            "Samples": n_samples,
            "Correct": n_correct,
            "Accuracy": f"{acc:.2%}",
        })

    summary_df = pd.DataFrame(summary_rows)
    results_df.to_csv(os.path.join(RESULTS_DIR, "task2_lesk_results.csv"), index=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "task2_lesk_summary.csv"), index=False)

    return results_df, summary_df


ALGORITHMS = {
    "Original Lesk": lambda s, w: wsd_module.original_lesk(s, w),
    "Adapted Lesk": lambda s, w: wsd_module.simple_lesk(s, w, hyperhypo=False),
    "Extended Lesk (hyperhypo)": lambda s, w: wsd_module.simple_lesk(s, w, hyperhypo=True),
    "Cosine Lesk": lambda s, w: wsd_module.cosine_lesk(s, w),
    "Path Similarity": lambda s, w: wsd_module.max_similarity(s, w, option="path"),
    "Info Content (Resnik)": lambda s, w: wsd_module.max_similarity(s, w, option="res"),
    "Wu-Palmer": lambda s, w: wsd_module.max_similarity(s, w, option="wup"),
    "Highest Lemma Count": lambda s, w: wsd_module.max_lemma_count(w),
}


def run_task4(dataset):
    """Evaluate all eight WSD algorithms on the full dataset."""
    records = []

    for algo_name, algo_fn in ALGORITHMS.items():
        print(f"  Evaluating: {algo_name} ...", end=" ", flush=True)
        t_start = time.time()

        reg_correct = reg_total = reg_eval = 0
        acr_correct = acr_total = acr_eval = 0

        for term, info in dataset.items():
            instances = info["instances"]
            res = evaluate_algorithm_on_term(algo_fn, term, instances)

            if info["term_type"] == "regular":
                reg_correct += res["correct"]
                reg_total += res["total"]
                reg_eval += res["evaluable"]
            else:
                acr_correct += res["correct"]
                acr_total += res["total"]
                acr_eval += res["evaluable"]

        elapsed = round(time.time() - t_start, 1)
        reg_acc = reg_correct / reg_eval if reg_eval > 0 else 0.0
        acr_acc = acr_correct / acr_eval if acr_eval > 0 else 0.0
        total_eval = reg_eval + acr_eval
        total_inst = reg_total + acr_total
        coverage = total_eval / total_inst if total_inst > 0 else 0.0

        records.append({
            "Algorithm": algo_name,
            "Regular Accuracy": round(reg_acc, 3),
            "Acronym Accuracy": round(acr_acc, 3),
            "Overall Accuracy": round((reg_correct + acr_correct) / total_eval, 3) if total_eval else 0.0,
            "Coverage": round(coverage, 3),
            "Execution Time (s)": elapsed,
        })
        print(f"reg={reg_acc:.2%}, acr={acr_acc:.2%}, time={elapsed}s")

    comparison_df = pd.DataFrame(records)
    comparison_df.to_csv(
        os.path.join(RESULTS_DIR, "task4_algorithm_comparison.csv"), index=False
    )
    return comparison_df


def run_task6(dataset, sample_size=5, seed=42):
    """Compare Wu-Palmer baseline against the combined semantic+phonetic method."""
    random.seed(seed)
    all_terms = list(dataset.keys())
    sample = random.sample(all_terms, min(sample_size, len(all_terms)))

    from pywsd.similarity import max_similarity

    records = []

    for term in sample:
        instances = dataset[term]["instances"]
        if not wn.synsets(term):
            continue

        wup_fn = lambda s, w: max_similarity(s, w, option="wup")
        wup_res = evaluate_algorithm_on_term(wup_fn, term, instances)

        combined_fn = lambda s, w: wsd_module.combined_wsd(
            s, w, phonetic_module, alpha=0.5
        )[0]
        comb_res = evaluate_algorithm_on_term(combined_fn, term, instances)

        records.append({
            "Term": term,
            "Term Type": dataset[term]["term_type"],
            "WuPalmer Accuracy": wup_res["accuracy"],
            "Combined Accuracy": comb_res["accuracy"],
            "WuPalmer Coverage": wup_res["coverage"],
            "Combined Coverage": comb_res["coverage"],
            "Instances": len(instances),
        })

    combined_df = pd.DataFrame(records)
    if not combined_df.empty:
        combined_df.to_csv(
            os.path.join(RESULTS_DIR, "task6_combined_results.csv"), index=False
        )
    return combined_df


def plot_algorithm_comparison(comparison_df):
    """Bar chart comparing algorithm accuracy across regular terms and acronyms."""
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(comparison_df))
    width = 0.35

    ax.bar(x - width / 2, comparison_df["Regular Accuracy"],
           width, label="Regular terms", color="#4e79a7")
    ax.bar(x + width / 2, comparison_df["Acronym Accuracy"],
           width, label="Acronyms", color="#f28e2b")

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Accuracy")
    ax.set_title("WSD Algorithm Accuracy on MSH-WSD Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Algorithm"], rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "task4_algorithm_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_combined_comparison(combined_df):
    """Bar chart comparing Wu-Palmer baseline against the combined method."""
    if combined_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(combined_df))
    width = 0.35

    ax.bar(x - width / 2, combined_df["WuPalmer Accuracy"],
           width, label="Wu-Palmer", color="#59a14f")
    ax.bar(x + width / 2, combined_df["Combined Accuracy"],
           width, label="Combined (WuP + Phonetic)", color="#e15759")

    ax.set_xlabel("Term")
    ax.set_ylabel("Accuracy")
    ax.set_title("Wu-Palmer vs Combined (Semantic + Phonetic) WSD Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(combined_df["Term"], rotation=20, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "task6_combined_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_phonetic_heatmap(words):
    """Generate and save a phonetic distance heatmap."""
    cleaned, matrix = phonetic_module.build_distance_matrix(words)

    fig, ax = plt.subplots(figsize=(max(6, len(cleaned)), max(5, len(cleaned) - 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Phonetic distance (edit distance on NYSIIS codes)")

    ax.set_xticks(range(len(cleaned)))
    ax.set_yticks(range(len(cleaned)))
    ax.set_xticklabels(cleaned, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cleaned, fontsize=8)

    for i in range(len(cleaned)):
        for j in range(len(cleaned)):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="black" if matrix[i, j] < matrix.max() * 0.7 else "white")

    ax.set_title("Phonetic Distance Matrix (NYSIIS + Edit Distance)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "task5_phonetic_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, cleaned, matrix


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = data_loader.load_or_create_dataset()
    stats = data_loader.get_dataset_stats(dataset)
    print("Dataset stats:", stats)

    print("\n" + "=" * 60)
    print("TASK 2 – Basic Lesk on 10 regular + 10 acronym sample")
    print("=" * 60)
    results_df, summary_df = run_task2(dataset)
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("TASK 5 – Phonetic heatmap")
    print("=" * 60)
    sentence = "I have been prescribed two important drugs today during my visit to clinic"
    from nltk.tokenize import word_tokenize
    tokens = [t for t in word_tokenize(sentence.lower()) if t.isalpha()]
    heatmap_path, cleaned, matrix = plot_phonetic_heatmap(tokens)
    # Save matrix as CSV
    pd.DataFrame(matrix, index=cleaned, columns=cleaned).to_csv(
        os.path.join(RESULTS_DIR, "task5_phonetic_matrix.csv")
    )
    print(f"Phonetic matrix saved. Heatmap: {heatmap_path}")

    print("\n" + "=" * 60)
    print("TASK 4 – All algorithms on full dataset")
    print("=" * 60)
    comparison_df = run_task4(dataset)
    chart_path = plot_algorithm_comparison(comparison_df)
    print(f"\nComparison chart saved: {chart_path}")
    print("\nAlgorithm comparison table:")
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("TASK 6 – Combined WSD evaluation")
    print("=" * 60)
    combined_df = run_task6(dataset, sample_size=5)
    if not combined_df.empty:
        combined_path = plot_combined_comparison(combined_df)
        print(f"Combined comparison chart saved: {combined_path}")
        print(combined_df.to_string(index=False))
    else:
        print("No evaluable terms for combined method.")

    print("\nAll evaluation results saved to results/")
