"""
Unit tests for the WSD project (Tasks 1–7).

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from nltk.corpus import wordnet as wn

import phonetic as phonetic_module
import wsd as wsd_module
import data_loader

# Test 1 – Basic Lesk

SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"
TARGET = "drug"

def test_basic_lesk_returns_result():
    """Lesk returns a synset or a clear error dict for the example sentence."""
    result = wsd_module.basic_lesk(SENTENCE, TARGET)
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert "synset" in result
    assert result["synset"]  # non-empty synset name

def test_basic_lesk_synset_is_valid_wordnet():
    """Predicted synset must be a real WordNet synset."""
    result = wsd_module.basic_lesk(SENTENCE, TARGET)
    if "error" in result:
        pytest.skip("Lesk returned no prediction")
    synset_name = result["synset"]
    synset = wn.synset(synset_name)
    assert synset is not None
    assert synset.definition()

# Test 2 – Multiple meanings (patient)

def test_lesk_patient_sentence():
    """Application handles 'patient' in a medical sentence."""
    sent = "The patient was admitted to the hospital for treatment."
    result = wsd_module.basic_lesk(sent, "patient")
    # Should return a result (even if the sense is noun or adjective)
    assert "error" not in result or "not found" not in result.get("error", "")

# Test 3 – Invalid target word

def test_lesk_invalid_target_word_not_in_wordnet():
    """'elephant' not in the sentence → error message about not found."""
    result = wsd_module.basic_lesk("The doctor prescribed medication.", "elephant")
    # elephant is in WordNet but not in sentence
    assert "error" in result

def test_lesk_target_not_in_wordnet_returns_error():
    """Nonsense word not in WordNet → clear error."""
    result = wsd_module.basic_lesk("The xyzzy was strange.", "xyzzy")
    assert "error" in result

# Test 4 – Algorithm comparison

def test_all_algorithms_execute():
    """All 8 algorithms run without raising exceptions."""
    results = wsd_module.run_all_algorithms(SENTENCE, TARGET)
    assert len(results) == 8

def test_algorithms_return_correct_keys():
    """Each algorithm result has required keys."""
    results = wsd_module.run_all_algorithms(SENTENCE, TARGET)
    for r in results:
        assert "algorithm" in r
        assert "synset" in r
        assert "definition" in r
        assert "elapsed_ms" in r

def test_algorithm_execution_times_recorded():
    """Execution times are non-negative numbers."""
    results = wsd_module.run_all_algorithms(SENTENCE, TARGET)
    for r in results:
        assert isinstance(r["elapsed_ms"], (int, float))
        assert r["elapsed_ms"] >= 0

# Test 5 – Phonetic distance matrix

def test_phonetic_matrix_is_square():
    """Distance matrix must be square."""
    words = ["drug", "clinic", "prescribed", "visit", "important"]
    cleaned, matrix = phonetic_module.build_distance_matrix(words)
    n = len(cleaned)
    assert matrix.shape == (n, n)

def test_phonetic_matrix_diagonal_is_zero():
    """Diagonal elements must be 0 (word vs itself)."""
    words = ["drug", "clinic", "prescribed"]
    cleaned, matrix = phonetic_module.build_distance_matrix(words)
    for i in range(len(cleaned)):
        assert matrix[i, i] == 0, f"Diagonal not zero at index {i}"

def test_phonetic_matrix_is_symmetric():
    """Distance matrix must be symmetric."""
    words = ["drug", "clinic", "prescribed", "visit"]
    cleaned, matrix = phonetic_module.build_distance_matrix(words)
    assert np.allclose(matrix, matrix.T), "Matrix is not symmetric"

def test_phonetic_matrix_is_nonnegative():
    """All distances must be non-negative."""
    words = ["drug", "clinic", "prescribed", "visit"]
    cleaned, matrix = phonetic_module.build_distance_matrix(words)
    assert (matrix >= 0).all(), "Negative distances found"

# Test 6 – Combined similarity bounds

def test_normalized_phonetic_similarity_bounds():
    """Normalized similarity must be in [0, 1]."""
    pairs = [
        ("drug", "drug"),
        ("drug", "clinic"),
        ("prescribed", "important"),
        ("visit", "hospital"),
    ]
    for w1, w2 in pairs:
        sim = phonetic_module.normalized_phonetic_similarity(w1, w2)
        assert 0.0 <= sim <= 1.0, f"Out-of-range similarity for ({w1}, {w2}): {sim}"

def test_normalized_phonetic_similarity_identical():
    """Same word should have similarity 1.0."""
    sim = phonetic_module.normalized_phonetic_similarity("drug", "drug")
    assert sim == 1.0

def test_combined_wsd_returns_synset_or_none():
    """Combined WSD returns a synset or None, never raises."""
    synset, score, details = wsd_module.combined_wsd(
        SENTENCE, TARGET, phonetic_module, alpha=0.5
    )
    # synset can be None if word not in WordNet; otherwise it must be valid
    if synset is not None:
        assert synset.name()
        assert 0.0 <= score <= 1.0

def test_combined_similarity_in_range():
    """Combined score must be in [0, 1]."""
    synset, score, details = wsd_module.combined_wsd(
        SENTENCE, TARGET, phonetic_module, alpha=0.5
    )
    assert 0.0 <= score <= 1.0

def test_combined_wsd_uses_both_components():
    """Score details must contain semantic, phonetic, and combined keys."""
    _, _, details = wsd_module.combined_wsd(
        SENTENCE, TARGET, phonetic_module, alpha=0.5
    )
    if details:
        for synset_name, scores in details.items():
            assert "semantic" in scores
            assert "phonetic" in scores
            assert "combined" in scores
            assert 0.0 <= scores["combined"] <= 1.0

# Test 7 – Dataset loading

def test_dataset_loads():
    """Dataset loads without error."""
    dataset = data_loader.load_or_create_dataset()
    assert len(dataset) > 0

def test_dataset_has_regular_and_acronym_terms():
    """Dataset must contain both regular terms and acronyms."""
    dataset = data_loader.load_or_create_dataset()
    regular, acronyms = data_loader.get_term_types(dataset)
    assert len(regular) >= 5, "Too few regular terms"
    assert len(acronyms) >= 5, "Too few acronyms"

def test_dataset_instances_have_required_fields():
    """Each instance must have sentence, cui, and cui_label."""
    dataset = data_loader.load_or_create_dataset()
    for term, info in dataset.items():
        for inst in info["instances"]:
            assert "sentence" in inst, f"Missing sentence for {term}"
            assert "cui" in inst, f"Missing cui for {term}"
            assert "cui_label" in inst, f"Missing cui_label for {term}"
            assert len(inst["sentence"]) > 10, f"Sentence too short for {term}"

def test_sample_terms_respects_seed():
    """Sampling with the same seed returns the same terms."""
    dataset = data_loader.load_or_create_dataset()
    s1_regular, s1_acronyms = data_loader.sample_terms(dataset, seed=42)
    s2_regular, s2_acronyms = data_loader.sample_terms(dataset, seed=42)
    assert s1_regular == s2_regular
    assert s1_acronyms == s2_acronyms

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
