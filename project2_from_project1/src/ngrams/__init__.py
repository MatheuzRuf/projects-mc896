"""N-gram feature extraction and k-nearest-neighbor relation classification."""

from project2_from_project1.src.ngrams.feature_matrix import (
    CandidateMetadata,
    NgramCandidateExample,
    NgramFeatureMatrix,
    NgramFeatureVectorizer,
)
from project2_from_project1.src.ngrams.knn_classifier import NgramKNNClassifier
from project2_from_project1.src.ngrams.model_selection import (
    NgramKNNSelectionError,
    build_final_training_partition,
    finalize_selected_ngram_knn,
    select_ngram_knn_from_folds,
)
from project2_from_project1.src.ngrams.ngrams_extractor import (
    NgramCounts,
    NgramFeature,
    build_ngram_vocabulary,
    extract_candidate_tokens,
    generate_ngrams,
    vectorize_ngrams,
)

__all__ = [
    "CandidateMetadata",
    "NgramCandidateExample",
    "NgramCounts",
    "NgramFeature",
    "NgramFeatureMatrix",
    "NgramFeatureVectorizer",
    "NgramKNNClassifier",
    "NgramKNNSelectionError",
    "build_ngram_vocabulary",
    "build_final_training_partition",
    "extract_candidate_tokens",
    "generate_ngrams",
    "finalize_selected_ngram_knn",
    "select_ngram_knn_from_folds",
    "vectorize_ngrams",
]
