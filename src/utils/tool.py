from typing import Dict, List

from sentence_transformers import SentenceTransformer, util


def search_top5_contents_from_query(
    query: str,
    contents: List[str],
    model: SentenceTransformer = None,
    model_name: str = "jhgan/ko-sroberta-multitask",
) -> List[str]:
    """
    Search top 5 contents from query using cosine similarity.

    Args:
        query: Query text to search for
        contents: List of content texts to search from
        model: Pre-loaded SentenceTransformer model (optional)
        model_name: Model name to use if model is not provided

    Returns:
        List of top 5 most similar contents
    """
    if not contents:
        return []

    # Load model if not provided
    if model is None:
        model = SentenceTransformer(model_name)

    # Encode query and contents
    query_embedding = model.encode(query, convert_to_tensor=True)
    content_embeddings = model.encode(contents, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(query_embedding, content_embeddings)[0]

    # Get top 5 indices
    top_k = min(5, len(contents))
    top_indices = similarities.argsort(descending=True)[:top_k]

    # Return top 5 contents
    return [contents[idx] for idx in top_indices.cpu().numpy()]


def get_content_from_code(code: str, codes: List[str], contents: List[str]) -> str:
    """
    Get content from code.

    Args:
        code: Achievement standard code (e.g., "10영03-04")
        codes: List of achievement standard codes
        contents: List of achievement standard contents (same length as codes)

    Returns:
        Content corresponding to the code, or "Invalid code" if not found
    """
    if len(codes) != len(contents):
        raise ValueError("codes and contents must have the same length")

    if code not in codes:
        return "Invalid code"

    idx = codes.index(code)
    return contents[idx]
