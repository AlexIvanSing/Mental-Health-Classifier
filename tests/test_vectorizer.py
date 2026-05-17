# ============================================================================
# Tests — Vectorizador TF-IDF
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
import pytest
import tempfile
import os
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from src.vectorizer import build_vectorizer
from src.utils import load_config

# --- FIXTURES ---

@pytest.fixture
def temp_config():
    config = {
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 0.95,
            "sublinear_tf": True,
            "max_features": 100
        }
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        yield path
    finally:
        os.remove(path)

@pytest.fixture
def sample_config(temp_config):
    return load_config(temp_config)

@pytest.fixture
def corpus():
    return [
        "I want to kill myself",
        "life is beautiful",
        "I cannot go on anymore",
        "today was a good day",
        "I feel hopeless and alone"
    ]

# --- PRUEBAS load_config ---

def test_load_config_returns_dict(temp_config):
    config = load_config(temp_config)
    assert isinstance(config, dict)

def test_load_config_has_vectorizer_key(temp_config):
    config = load_config(temp_config)
    assert "vectorizer" in config

def test_load_config_vectorizer_has_required_keys(temp_config):
    config = load_config(temp_config)
    required_keys = ["ngram_range", "min_df", "max_df", "sublinear_tf", "max_features"]
    for key in required_keys:
        assert key in config["vectorizer"]

def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")

# --- PRUEBAS build_vectorizer ---

def test_build_vectorizer_returns_tfidf_instance(sample_config):
    vectorizer = build_vectorizer(sample_config)
    assert isinstance(vectorizer, TfidfVectorizer)

def test_build_vectorizer_ngram_range(sample_config):
    vectorizer = build_vectorizer(sample_config)
    assert vectorizer.ngram_range == (1, 2)

def test_build_vectorizer_ngram_range_is_tuple(sample_config):
    vectorizer = build_vectorizer(sample_config)
    assert isinstance(vectorizer.ngram_range, tuple)

def test_build_vectorizer_sublinear_tf(sample_config):
    vectorizer = build_vectorizer(sample_config)
    assert vectorizer.sublinear_tf is True

def test_build_vectorizer_max_features(sample_config):
    vectorizer = build_vectorizer(sample_config)
    assert vectorizer.max_features == 100

# --- PRUEBAS fit + transform (corpus toy) ---

def test_fit_transform_returns_correct_rows(sample_config, corpus):
    vectorizer = build_vectorizer(sample_config)
    matrix = vectorizer.fit_transform(corpus)
    assert matrix.shape[0] == len(corpus)

def test_fit_transform_matrix_is_sparse(sample_config, corpus):
    from scipy.sparse import issparse
    vectorizer = build_vectorizer(sample_config)
    matrix = vectorizer.fit_transform(corpus)
    assert issparse(matrix)

def test_fit_transform_max_features_respected(sample_config, corpus):
    vectorizer = build_vectorizer(sample_config)
    matrix = vectorizer.fit_transform(corpus)
    assert matrix.shape[1] <= sample_config["vectorizer"]["max_features"]

def test_fit_transform_no_empty_matrix(sample_config, corpus):
    vectorizer = build_vectorizer(sample_config)
    matrix = vectorizer.fit_transform(corpus)
    assert matrix.nnz > 0