import fuzzy
import jellyfish
import numpy as np

_dmetaphone = fuzzy.DMetaphone()

def phonetic_code(word):
    """Return the NYSIIS phonetic code for word using the fuzzy library."""
    if not word or not word.isalpha():
        return None
    try:
        code = fuzzy.nysiis(word.upper())
        if code:
            return code
    except Exception:
        pass
    # fallback to Double Metaphone
    try:
        result = _dmetaphone(word.upper())
        if result and result[0]:
            return result[0].decode("ascii")
    except Exception:
        pass
    return None


def soundex_code(word):
    """Return jellyfish Soundex code for display purposes."""
    try:
        return jellyfish.soundex(word)
    except Exception:
        return None


def edit_distance(s1, s2):
    """Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[len(s2)]


def phonetic_distance(word1, word2):
    """Edit distance between the NYSIIS codes of two words."""
    code1 = phonetic_code(word1)
    code2 = phonetic_code(word2)
    if code1 is None or code2 is None:
        return edit_distance(word1.lower(), word2.lower())
    return edit_distance(code1, code2)


def normalized_phonetic_similarity(word1, word2):
    """Phonetic similarity normalized to [0, 1], where 1 means identical."""
    code1 = phonetic_code(word1) or word1.lower()
    code2 = phonetic_code(word2) or word2.lower()
    dist = edit_distance(code1, code2)
    max_len = max(len(code1), len(code2), 1)
    return max(0.0, 1.0 - dist / max_len)


def build_distance_matrix(words):
    """Build a pairwise phonetic distance matrix for a list of words."""
    cleaned = [w for w in words if w.isalpha()]
    n = len(cleaned)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = phonetic_distance(cleaned[i], cleaned[j])
            matrix[i, j] = d
            matrix[j, i] = d
    return cleaned, matrix


def get_phonetic_codes(words):
    """Return {word: {'nysiis': ..., 'soundex': ...}} for each word."""
    return {w: {"nysiis": phonetic_code(w), "soundex": soundex_code(w)} for w in words}


if __name__ == "__main__":
    from nltk.tokenize import word_tokenize

    sentence = "I have been prescribed two important drugs today during my visit to clinic"
    words = [t for t in word_tokenize(sentence.lower()) if t.isalpha()]

    print("Phonetic codes (NYSIIS | Soundex):")
    for w, c in get_phonetic_codes(words).items():
        print(f"  {w:15s} -> {c['nysiis']:15s}  {c['soundex']}")

    cleaned, matrix = build_distance_matrix(words)
    print(f"\nMatrix shape: {matrix.shape}, diagonal zeros: {all(matrix[i,i]==0 for i in range(len(cleaned)))}")

    for w1, w2 in [("drugs", "drug"), ("visit", "clinic")]:
        print(f"  {w1} vs {w2}: dist={phonetic_distance(w1,w2)}, sim={normalized_phonetic_similarity(w1,w2):.3f}")
