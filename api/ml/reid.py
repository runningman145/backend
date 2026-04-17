"""
Re-identification (ReID) helpers.
Cosine similarity and other matching metrics.
"""
import numpy as np
from typing import List, Union, Optional


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: normalized embedding vector (numpy array)
        embedding2: normalized embedding vector (numpy array)
    
    Returns:
        Similarity score (0-1) where 1 means identical direction
    
    Examples:
        >>> emb1 = np.array([1, 0, 0])
        >>> emb2 = np.array([1, 0, 0])
        >>> cosine_similarity(emb1, emb2)
        1.0
        
        >>> emb1 = np.array([1, 0, 0])
        >>> emb2 = np.array([0, 1, 0])
        >>> cosine_similarity(emb1, emb2)
        0.0
    """
    # Ensure inputs are numpy arrays
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)
    
    # Calculate dot product and norms
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Avoid division by zero
    denominator = norm1 * norm2 + 1e-8
    
    similarity = dot_product / denominator
    
    # Clamp to valid range due to floating point errors
    similarity = np.clip(similarity, -1.0, 1.0)
    
    # Convert to [0, 1] range if needed (for similarity, not distance)
    # Cosine similarity naturally ranges from -1 to 1
    # For ReID, we usually want 0-1, so map negative to 0
    if similarity < 0:
        similarity = 0.0
    
    return float(similarity)


def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine distance between two embeddings.
    
    Args:
        embedding1: normalized embedding vector
        embedding2: normalized embedding vector
    
    Returns:
        Distance score (0-2) where 0 means identical direction
    
    Examples:
        >>> emb1 = np.array([1, 0, 0])
        >>> emb2 = np.array([1, 0, 0])
        >>> cosine_distance(emb1, emb2)
        0.0
    """
    similarity = cosine_similarity(embedding1, embedding2)
    return 1.0 - similarity


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        embedding1: embedding vector
        embedding2: embedding vector
    
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(embedding1 - embedding2))


def euclidean_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Convert Euclidean distance to similarity score.
    
    Args:
        embedding1: embedding vector
        embedding2: embedding vector
    
    Returns:
        Similarity score (0-1) where 1 means identical
    """
    distance = euclidean_distance(embedding1, embedding2)
    # Convert distance to similarity using exponential decay
    similarity = np.exp(-distance)
    return float(similarity)


def batch_cosine_similarity(query_embedding: np.ndarray, gallery_embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Calculate cosine similarity between a query embedding and multiple gallery embeddings.
    
    Args:
        query_embedding: Single query embedding vector
        gallery_embeddings: List of gallery embedding vectors
    
    Returns:
        Array of similarity scores
    
    Examples:
        >>> query = np.array([1, 0, 0])
        >>> gallery = [np.array([1, 0, 0]), np.array([0, 1, 0])]
        >>> batch_cosine_similarity(query, gallery)
        array([1.0, 0.0])
    """
    if not gallery_embeddings:
        return np.array([])
    
    # Stack gallery embeddings into matrix
    gallery_matrix = np.stack(gallery_embeddings)
    
    # Normalize query if needed
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    
    # Normalize gallery if needed (assumes already normalized, but safe to normalize again)
    gallery_norms = np.linalg.norm(gallery_matrix, axis=1, keepdims=True)
    gallery_normalized = gallery_matrix / (gallery_norms + 1e-8)
    
    # Compute all similarities at once
    similarities = np.dot(gallery_normalized, query_norm)
    
    # Clamp to valid range
    similarities = np.clip(similarities, -1.0, 1.0)
    similarities[similarities < 0] = 0.0
    
    return similarities


def find_top_k_matches(query_embedding: np.ndarray, 
                       gallery_embeddings: List[np.ndarray], 
                       gallery_ids: Optional[List] = None,
                       k: int = 5,
                       threshold: float = 0.5) -> List[dict]:
    """
    Find top K matches for a query embedding from a gallery.
    
    Args:
        query_embedding: Query embedding vector
        gallery_embeddings: List of gallery embeddings
        gallery_ids: Optional list of IDs corresponding to gallery embeddings
        k: Number of top matches to return
        threshold: Minimum similarity threshold
    
    Returns:
        List of dictionaries containing match information
    
    Examples:
        >>> query = np.array([1, 0, 0])
        >>> gallery = [np.array([1, 0, 0]), np.array([0.9, 0.1, 0]), np.array([0, 1, 0])]
        >>> ids = ['car1', 'car2', 'car3']
        >>> find_top_k_matches(query, gallery, ids, k=2)
        [{'id': 'car1', 'similarity': 1.0, 'rank': 1}, {'id': 'car2', 'similarity': 0.99, 'rank': 2}]
    """
    if not gallery_embeddings:
        return []
    
    # Calculate all similarities
    similarities = batch_cosine_similarity(query_embedding, gallery_embeddings)
    
    # Create list of (similarity, index, id) tuples
    matches = []
    for idx, sim in enumerate(similarities):
        if sim >= threshold:
            match_info = {
                'similarity': float(sim),
                'index': idx,
                'rank': 0  # Will set after sorting
            }
            if gallery_ids and idx < len(gallery_ids):
                match_info['id'] = gallery_ids[idx]
            matches.append((sim, idx, match_info))
    
    # Sort by similarity descending
    matches.sort(key=lambda x: x[0], reverse=True)
    
    # Take top K and add rank
    top_matches = []
    for rank, (sim, idx, match_info) in enumerate(matches[:k], 1):
        match_info['rank'] = rank
        top_matches.append(match_info)
    
    return top_matches


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize an embedding vector to unit length.
    
    Args:
        embedding: Input embedding vector
    
    Returns:
        Normalized embedding vector
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def batch_normalize_embeddings(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize multiple embedding vectors.
    
    Args:
        embeddings: List of embedding vectors
    
    Returns:
        List of normalized embedding vectors
    """
    return [normalize_embedding(emb) for emb in embeddings]


# Alias for backward compatibility
def cosine_similarity_percent(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity as percentage (0-100).
    
    Args:
        embedding1: normalized embedding vector
        embedding2: normalized embedding vector
    
    Returns:
        Similarity percentage (0-100)
    """
    return cosine_similarity(embedding1, embedding2) * 100


# Utility function to validate embeddings
def validate_embedding(embedding: np.ndarray, expected_dim: Optional[int] = None) -> bool:
    """
    Validate that an embedding has the expected properties.
    
    Args:
        embedding: Embedding vector to validate
        expected_dim: Optional expected dimension
    
    Returns:
        True if embedding is valid
    """
    if not isinstance(embedding, np.ndarray):
        return False
    
    if embedding.ndim != 1:
        return False
    
    if expected_dim is not None and embedding.shape[0] != expected_dim:
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        return False
    
    return True