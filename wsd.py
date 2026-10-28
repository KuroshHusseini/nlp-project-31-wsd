import time
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from pywsd.lesk import original_lesk, simple_lesk, adapted_lesk, cosine_lesk
from pywsd.similarity import max_similarity
from pywsd import max_lemma_count

_lemmatizer = WordNetLemmatizer()


def to_synset(result):
    """Normalize a pywsd result (Synset, name string, tuple/list, dict) to a Synset or None."""
    if result is None:
        return None
    if hasattr(result, "definition") and hasattr(result, "name"):
        return result
    if isinstance(result, str):
        try:
            return wn.synset(result)
        except Exception:
            return None
    if isinstance(result, dict):
        result = list(result.items())
    if isinstance(result, (tuple, list)):
        for item in result:
            syn = to_synset(item)
            if syn is not None:
                return syn
    return None


def basic_lesk(sentence, target_word, pos=None):
    """Run NLTK Lesk on the sentence and return sense info or an error dict."""
    tokens = word_tokenize(sentence.lower())
    lemmatized = {_lemmatizer.lemmatize(t) for t in tokens}

    if target_word.lower() not in tokens and target_word.lower() not in lemmatized:
        return {"error": f"'{target_word}' not found in the sentence."}

    if not wn.synsets(target_word, pos=pos):
        return {"error": f"'{target_word}' has no WordNet entries."}

    sense = lesk(tokens, target_word, pos=pos)
    if sense is None:
        return {"error": "Lesk returned no prediction for this word/context."}

    return {
        "algorithm": "NLTK Lesk",
        "sentence": sentence,
        "target_word": target_word,
        "synset": sense.name(),
        "definition": sense.definition(),
        "examples": sense.examples(),
        "pos": sense.pos(),
        "lexname": sense.lexname(),
        "all_synsets": [s.name() for s in wn.synsets(target_word, pos=pos)],
        "explanation": (
            "Lesk selects the synset whose definition and examples share the most "
            "words with the context sentence."
        ),
    }


def format_lesk_output(result):
    if "error" in result:
        return f"Error: {result['error']}"
    lines = [
        f"Sentence:\n  {result['sentence']}",
        f"\nTarget word:\n  {result['target_word']}",
        f"\nAlgorithm:\n  {result['algorithm']}",
        f"\nPredicted sense:\n  {result['synset']}",
        f"\nDefinition:\n  {result['definition']}",
    ]
    if result["examples"]:
        lines.append("\nWordNet examples:")
        for ex in result["examples"]:
            lines.append(f"  - {ex}")
    lines.append(f"\nExplanation:\n  {result['explanation']}")
    return "\n".join(lines)


def run_all_algorithms(sentence, target_word):
    """Return results for all eight WSD algorithms as a list of dicts."""
    results = []

    def record(name, synset, elapsed):
        synset = to_synset(synset)
        if synset is None:
            syn_name = None
            syn_def = "No prediction returned."
        else:
            syn_name = synset.name()
            syn_def = synset.definition()
        results.append({
            "algorithm": name,
            "synset": syn_name,
            "definition": syn_def,
            "elapsed_ms": round(elapsed * 1000, 1),
        })

    for name, fn in [
        ("Original Lesk",              lambda: original_lesk(sentence, target_word)),
        ("Adapted Lesk",               lambda: simple_lesk(sentence, target_word, hyperhypo=False)),
        ("Extended Lesk (hyperhypo)",  lambda: simple_lesk(sentence, target_word, hyperhypo=True)),
        ("Cosine Lesk",                lambda: cosine_lesk(sentence, target_word)),
        ("Path Similarity",            lambda: max_similarity(sentence, target_word, option="path")),
        ("Information Content (Resnik)", lambda: max_similarity(sentence, target_word, option="res")),
        ("Wu-Palmer Similarity",       lambda: max_similarity(sentence, target_word, option="wup")),
        ("Highest Lemma Count",        lambda: max_lemma_count(target_word)),
    ]:
        t0 = time.time()
        try:
            s = fn()
        except Exception:
            s = None
        record(name, s, time.time() - t0)

    return results


def _wup_sim(a, b):
    if a is None or b is None:
        return 0.0
    try:
        v = a.wup_similarity(b)
        return v if v is not None else 0.0
    except Exception:
        return 0.0


def combined_wsd(sentence, target_word, phonetic_module, alpha=0.5):
    """
    Disambiguate target_word using a combination of Wu-Palmer semantic
    similarity and phonetic similarity (weighted by alpha).

    For each candidate synset, context tokens are compared against the
    synset definition words using both metrics. The synset with the
    highest combined score is returned.
    """
    tokens = word_tokenize(sentence.lower())
    context = [t for t in tokens if t.isalpha() and t != target_word.lower()]

    candidates = wn.synsets(target_word)
    if not candidates:
        return None, 0.0, {}

    best_synset, best_score = None, -1.0
    details = {}

    for synset in candidates:
        def_words = word_tokenize(synset.definition().lower())
        for ex in synset.examples():
            def_words += word_tokenize(ex.lower())
        def_words = [w for w in def_words if w.isalpha()]
        def_synsets = [
            s for dw in set(def_words)
            for s in wn.synsets(dw)
            if s.pos() == synset.pos()
        ][:20]

        sem_scores, phon_scores = [], []

        for tok in context[:15]:
            ctx_synsets = wn.synsets(tok)
            if not ctx_synsets:
                continue
            sem = max((_wup_sim(cs, ds) for cs in ctx_synsets for ds in def_synsets), default=0.0)
            phon = max(
                (phonetic_module.normalized_phonetic_similarity(tok, dw) for dw in set(def_words) if dw),
                default=0.0,
            )
            sem_scores.append(sem)
            phon_scores.append(phon)

        s_sem = sum(sem_scores) / len(sem_scores) if sem_scores else 0.0
        s_phon = sum(phon_scores) / len(phon_scores) if phon_scores else 0.0
        s_combined = alpha * s_sem + (1 - alpha) * s_phon

        details[synset.name()] = {
            "semantic": round(s_sem, 4),
            "phonetic": round(s_phon, 4),
            "combined": round(s_combined, 4),
        }
        if s_combined > best_score:
            best_score, best_synset = s_combined, synset

    return best_synset, round(best_score, 4), details


if __name__ == "__main__":
    SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"
    TARGET = "drug"

    print("=" * 60)
    print("TASK 1 – Basic Lesk")
    print("=" * 60)
    print(format_lesk_output(basic_lesk(SENTENCE, TARGET)))

    print("\n" + "=" * 60)
    print("TASK 3 – All WSD Algorithms")
    print("=" * 60)
    for r in run_all_algorithms(SENTENCE, TARGET):
        print(f"\n{r['algorithm']}")
        print(f"  Synset    : {r['synset'] or 'No prediction'}")
        if r["synset"]:
            print(f"  Definition: {r['definition'][:80]}...")
        print(f"  Time      : {r['elapsed_ms']} ms")
