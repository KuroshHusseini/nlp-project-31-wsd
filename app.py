import os
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@st.cache_resource(show_spinner="Downloading language resources…")
def _ensure_nltk_resources():
    import nltk
    resources = [
        "punkt_tab", "punkt", "wordnet", "omw-1.4",
        "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger",
        "wordnet_ic", "semcor", "stopwords",
    ]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

_ensure_nltk_resources()

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk

import data_loader
import phonetic as phonetic_module
import wsd as wsd_module

st.set_page_config(
    page_title="WSD – Medical Context",
    page_icon="🔬",
    layout="wide",
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

@st.cache_resource(show_spinner="Loading dataset…")
def load_dataset():
    return data_loader.load_or_create_dataset()

def _result_file(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)

def _plot_file(name: str) -> str:
    return os.path.join(PLOTS_DIR, name)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "About",
        "WSD Analyzer",
        "Algorithm Comparison",
        "Phonetic Analysis",
        "Evaluation Results",
    ],
)

# Page 1: WSD Analyzer

if page == "WSD Analyzer":
    st.title("Word Sense Disambiguation – Analyzer")
    st.markdown(
        "Enter a sentence and a target word to see which sense the selected "
        "algorithm assigns to the word in context."
    )

    with st.form("wsd_form"):
        col1, col2 = st.columns([3, 1])
        sentence = col1.text_area(
            "Sentence",
            value="I have been prescribed two important drugs today during my visit to clinic",
            height=80,
        )
        target_word = col2.text_input("Target word", value="drug")

        algorithm = st.selectbox(
            "Algorithm",
            [
                "NLTK Lesk",
                "Original Lesk",
                "Adapted Lesk",
                "Extended Lesk (hyperhypo)",
                "Cosine Lesk",
                "Path Similarity",
                "Information Content (Resnik)",
                "Wu-Palmer Similarity",
                "Highest Lemma Count",
                "Combined (Wu-Palmer + Phonetic)",
            ],
        )
        submitted = st.form_submit_button("Analyze")

    if submitted:

        sentence = sentence.strip()
        target_word = target_word.strip().lower()

        if not sentence:
            st.error("Please enter a sentence.")
            st.stop()
        if not target_word:
            st.error("Please enter a target word.")
            st.stop()

        tokens = word_tokenize(sentence.lower())
        if target_word not in tokens:
            st.warning(
                f"'{target_word}' was not found in the sentence. "
                "Attempting disambiguation anyway (the word may appear in a different form)."
            )

        if not wn.synsets(target_word):
            st.error(f"'{target_word}' has no entries in WordNet. Cannot disambiguate.")
            st.stop()

        predicted = None
        score = 0.0
        details = {}
        t0 = time.time()

        try:
            if algorithm == "NLTK Lesk":
                predicted = lesk(tokens, target_word)
            elif algorithm == "Original Lesk":
                from pywsd.lesk import original_lesk
                predicted = original_lesk(sentence, target_word)
            elif algorithm == "Adapted Lesk":
                from pywsd.lesk import simple_lesk
                predicted = simple_lesk(sentence, target_word, hyperhypo=False)
            elif algorithm == "Extended Lesk (hyperhypo)":
                from pywsd.lesk import simple_lesk
                predicted = simple_lesk(sentence, target_word, hyperhypo=True)
            elif algorithm == "Cosine Lesk":
                from pywsd.lesk import cosine_lesk
                predicted = cosine_lesk(sentence, target_word)
            elif algorithm == "Path Similarity":
                from pywsd.similarity import max_similarity
                predicted = max_similarity(sentence, target_word, option="path")
            elif algorithm == "Information Content (Resnik)":
                from pywsd.similarity import max_similarity
                predicted = max_similarity(sentence, target_word, option="res")
            elif algorithm == "Wu-Palmer Similarity":
                from pywsd.similarity import max_similarity
                predicted = max_similarity(sentence, target_word, option="wup")
            elif algorithm == "Highest Lemma Count":
                from pywsd import max_lemma_count
                predicted = max_lemma_count(target_word)
            elif algorithm == "Combined (Wu-Palmer + Phonetic)":
                predicted, score, details = wsd_module.combined_wsd(
                    sentence, target_word, phonetic_module, alpha=0.5
                )
        except Exception as exc:
            st.error(f"Algorithm error: {exc}")
            st.stop()

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        if predicted is None:
            st.warning(
                "The algorithm returned no prediction. This can happen when "
                "the word has senses but none overlaps with the context, or "
                "when the algorithm cannot compare the word to context synsets."
            )
        else:
            st.success("Disambiguation complete")

            col_a, col_b = st.columns(2)
            col_a.metric("Predicted sense", predicted.name())
            col_b.metric("Execution time", f"{elapsed_ms} ms")

            st.subheader("Sense details")
            st.markdown(f"**Algorithm:** {algorithm}")
            st.markdown(f"**Synset:** `{predicted.name()}`")
            st.markdown(f"**POS tag:** {predicted.pos()}")
            st.markdown(f"**Lexical name:** {predicted.lexname()}")
            st.markdown(f"**Definition:** {predicted.definition()}")

            examples = predicted.examples()
            if examples:
                st.markdown("**WordNet example sentences:**")
                for ex in examples:
                    st.markdown(f"- _{ex}_")

            # All candidate synsets
            with st.expander("All WordNet synsets for this word"):
                candidate_synsets = [s for s in wn.synsets(target_word) if s is not None]
                for s in candidate_synsets:
                    if s is None:
                        continue
                    marker = "✅" if s == predicted else "  "
                    st.markdown(f"{marker} `{s.name()}` – {s.definition()}")

            # Combined score breakdown
            if algorithm == "Combined (Wu-Palmer + Phonetic)" and details:
                with st.expander("Combined score breakdown"):
                    score_df = pd.DataFrame(
                        [
                            {"Synset": k, **v}
                            for k, v in details.items()
                        ]
                    ).sort_values("combined", ascending=False)
                    st.dataframe(score_df, use_container_width=True)

# Page 2: Algorithm Comparison

elif page == "Algorithm Comparison":
    st.title("Compare WSD Algorithms")
    st.markdown(
        "Run multiple algorithms on the same sentence and compare their predictions."
    )

    with st.form("compare_form"):
        sentence = st.text_area(
            "Sentence",
            value="I have been prescribed two important drugs today during my visit to clinic",
            height=80,
        )
        target_word = st.text_input("Target word", value="drug")
        run_all = st.form_submit_button("Run all algorithms")

    if run_all:
        sentence = sentence.strip()
        target_word = target_word.strip().lower()

        if not sentence or not target_word:
            st.error("Please fill in both fields.")
            st.stop()
        if not wn.synsets(target_word):
            st.error(f"'{target_word}' has no WordNet entries.")
            st.stop()

        with st.spinner("Running all algorithms…"):
            results = wsd_module.run_all_algorithms(sentence, target_word)

        # Also add combined
        t0 = time.time()
        comb_synset, comb_score, _ = wsd_module.combined_wsd(
            sentence, target_word, phonetic_module, alpha=0.5
        )
        comb_elapsed = round((time.time() - t0) * 1000, 1)
        results.append({
            "algorithm": "Combined (Wu-Palmer + Phonetic)",
            "synset": comb_synset.name() if comb_synset else None,
            "definition": comb_synset.definition() if comb_synset else "No prediction.",
            "elapsed_ms": comb_elapsed,
        })

        rows = []
        for r in results:
            rows.append({
                "Algorithm": r["algorithm"],
                "Predicted Synset": r["synset"] or "—",
                "Definition": r["definition"][:90] + "…" if len(r.get("definition", "")) > 90 else r.get("definition", ""),
                "Time (ms)": r["elapsed_ms"],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# Page 3: Phonetic Analysis

elif page == "Phonetic Analysis":
    st.title("Phonetic Similarity Analysis")
    st.markdown(
        "Generate phonetic representations (NYSIIS via the `fuzzy` library) "
        "and a pairwise phonetic distance matrix for any sentence."
    )

    with st.form("phonetic_form"):
        sentence = st.text_area(
            "Sentence",
            value="I have been prescribed two important drugs today during my visit to clinic",
            height=80,
        )
        analyze = st.form_submit_button("Analyze")

    if analyze:
        sentence = sentence.strip()
        if not sentence:
            st.error("Please enter a sentence.")
            st.stop()

        tokens = word_tokenize(sentence.lower())
        words = [t for t in tokens if t.isalpha()]

        if len(words) < 2:
            st.error("Need at least two words for a distance matrix.")
            st.stop()

        # Phonetic codes table
        st.subheader("Phonetic Codes")
        codes = phonetic_module.get_phonetic_codes(words)
        code_rows = [
            {"Word": w, "NYSIIS (fuzzy)": v["nysiis"] or "N/A", "Soundex": v["soundex"] or "N/A"}
            for w, v in codes.items()
        ]
        st.dataframe(pd.DataFrame(code_rows), use_container_width=True)

        st.markdown(
            "**NYSIIS** (New York State Identification and Intelligence System) "
            "converts words into phonetic keys using the `fuzzy` library. "
            "**Soundex** provides a cross-check using the `jellyfish` library. "
            "Words with the same code sound similar."
        )

        # Distance matrix
        st.subheader("Phonetic Distance Matrix")
        cleaned, matrix = phonetic_module.build_distance_matrix(words)
        matrix_df = pd.DataFrame(matrix, index=cleaned, columns=cleaned)
        st.dataframe(matrix_df, use_container_width=True)

        st.markdown(
            "Values show Levenshtein edit distance between NYSIIS phonetic codes. "
            "**0 = phonetically identical**, higher = more different."
        )

        # Heatmap
        st.subheader("Phonetic Distance Heatmap")
        fig, ax = plt.subplots(figsize=(max(6, len(cleaned)), max(5, len(cleaned) - 1)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Edit distance on NYSIIS codes")
        ax.set_xticks(range(len(cleaned)))
        ax.set_yticks(range(len(cleaned)))
        ax.set_xticklabels(cleaned, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cleaned, fontsize=8)
        for i in range(len(cleaned)):
            for j in range(len(cleaned)):
                ax.text(j, i, str(matrix[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="black" if matrix[i, j] < matrix.max() * 0.7 else "white")
        ax.set_title("Phonetic Distance Matrix")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Pairwise similarity for selected pair
        st.subheader("Word pair similarity")
        col1, col2 = st.columns(2)
        w1 = col1.selectbox("Word 1", cleaned, index=0)
        w2 = col2.selectbox("Word 2", cleaned, index=min(1, len(cleaned) - 1))
        dist = phonetic_module.phonetic_distance(w1, w2)
        sim = phonetic_module.normalized_phonetic_similarity(w1, w2)
        c1, c2, c3 = st.columns(3)
        c1.metric("Phonetic distance", dist)
        c2.metric("Normalized similarity", f"{sim:.3f}")
        c3.metric("NYSIIS codes", f"{phonetic_module.phonetic_code(w1)} vs {phonetic_module.phonetic_code(w2)}")

# Page 4: Evaluation Results

elif page == "Evaluation Results":
    st.title("Evaluation Results")
    st.markdown(
        "Pre-computed results from running evaluation scripts on the MSH-WSD dataset. "
        "Run `python evaluation.py` to regenerate."
    )

    # Task 2
    st.header("Task 2 – NLTK Lesk on Sample")
    f2 = _result_file("task2_lesk_summary.csv")
    if os.path.exists(f2):
        df2 = pd.read_csv(f2)
        st.table(df2)
        st.caption(
            "Accuracy is measured using word-overlap between predicted WordNet definitions "
            "and UMLS sense labels. Coverage excludes words absent from WordNet."
        )
    else:
        st.info("Run `python evaluation.py` to generate results.")

    # Task 2 detail
    f2d = _result_file("task2_lesk_results.csv")
    if os.path.exists(f2d):
        with st.expander("Full per-instance results"):
            st.dataframe(pd.read_csv(f2d), use_container_width=True)

    # Task 4
    st.header("Task 4 – Algorithm Comparison")
    f4 = _result_file("task4_algorithm_comparison.csv")
    if os.path.exists(f4):
        df4 = pd.read_csv(f4)
        st.dataframe(
            df4.style.format({
                "Regular Accuracy": "{:.1%}",
                "Acronym Accuracy": "{:.1%}",
                "Overall Accuracy": "{:.1%}",
                "Coverage": "{:.1%}",
            }),
            use_container_width=True,
        )
        chart_path = _plot_file("task4_algorithm_comparison.png")
        if os.path.exists(chart_path):
            st.image(chart_path, caption="Algorithm accuracy comparison")
    else:
        st.info("Run `python evaluation.py` to generate results.")

    # Task 5
    st.header("Task 5 – Phonetic Distance Matrix")
    heatmap_path = _plot_file("task5_phonetic_heatmap.png")
    if os.path.exists(heatmap_path):
        st.image(heatmap_path, caption="Phonetic distance heatmap for the example sentence")
        f5 = _result_file("task5_phonetic_matrix.csv")
        if os.path.exists(f5):
            with st.expander("Numeric matrix"):
                st.dataframe(pd.read_csv(f5, index_col=0), use_container_width=True)
    else:
        st.info("Run `python evaluation.py` to generate results.")

    # Task 6
    st.header("Task 6 – Combined Semantic + Phonetic WSD")
    f6 = _result_file("task6_combined_results.csv")
    if os.path.exists(f6):
        df6 = pd.read_csv(f6)
        st.dataframe(
            df6.style.format({
                "WuPalmer Accuracy": "{:.1%}",
                "Combined Accuracy": "{:.1%}",
                "WuPalmer Coverage": "{:.1%}",
                "Combined Coverage": "{:.1%}",
            }),
            use_container_width=True,
        )
        chart6 = _plot_file("task6_combined_comparison.png")
        if os.path.exists(chart6):
            st.image(chart6, caption="Wu-Palmer vs Combined method accuracy")

        wup_mean = df6["WuPalmer Accuracy"].mean()
        comb_mean = df6["Combined Accuracy"].mean()
        col1, col2 = st.columns(2)
        col1.metric("Avg. Wu-Palmer accuracy", f"{wup_mean:.1%}")
        col2.metric("Avg. Combined accuracy", f"{comb_mean:.1%}",
                    delta=f"{comb_mean - wup_mean:+.1%}")

        st.caption(
            "In this evaluation both methods perform similarly because the word-overlap "
            "scoring function is the same underlying evaluation. The combined method "
            "adds phonetic context weighting (α=0.5) which benefits words with strong "
            "phonetic distinctions between senses, but may not improve performance on "
            "short biomedical acronyms where WordNet coverage is limited."
        )
    else:
        st.info("Run `python evaluation.py` to generate results.")

# Page 5: About

elif page == "About":
    st.title("About this Project")
    st.markdown("""
## Word Sense Disambiguation in Medical Context

**University of Oulu – NLP and Text Mining Course, 2026**
Project 31

### What is Word Sense Disambiguation?

Many words carry multiple meanings depending on context.  In biomedical text this
ambiguity is especially common: "cold" can mean *common cold* or *cold temperature*;
"MS" can mean *multiple sclerosis*, *mitral stenosis*, or *mass spectrometry*.

Word Sense Disambiguation (WSD) is the NLP task of automatically determining which
sense of an ambiguous word is intended in a given context.

### Implemented approaches

| Task | Method |
|------|--------|
| 1 | NLTK Lesk (context-definition overlap) |
| 3 | Original Lesk, Adapted Lesk, Extended Lesk, Cosine Lesk, Path similarity, IC similarity, Wu-Palmer, Highest lemma count |
| 5 | Phonetic similarity via NYSIIS (fuzzy library) + edit distance |
| 6 | Combined Wu-Palmer + phonetic (α = 0.5) |

### Dataset

A curated biomedical WSD dataset based on the **MSH-WSD** terminology
(Stevenson et al., 2010) with UMLS Concept Unique Identifier (CUI) annotations.
Includes 27 ambiguous terms (17 regular, 10 acronyms) with 259 annotated instances.

### Evaluation note

MSH-WSD sense labels are UMLS CUIs, not WordNet synset IDs.
Accuracy is estimated using word-overlap between predicted WordNet definitions and
UMLS concept labels — an approximation that tends to underestimate true accuracy.
Coverage is reported separately.

### Libraries used

`nltk`, `pywsd`, `fuzzy`, `jellyfish`, `pandas`, `numpy`, `matplotlib`, `streamlit`
""")

    dataset = load_dataset()
    stats = data_loader.get_dataset_stats(dataset)
    st.subheader("Dataset statistics")
    st.json(stats)
