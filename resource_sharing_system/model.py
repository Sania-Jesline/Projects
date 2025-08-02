import re
from math import sqrt

def semantic_search_from_firebase(query, firebase_resources, top_n=5):
    documents = []
    metadata = []

    # Flatten and preprocess data
    for user_id, resources in firebase_resources.items():
        for res_id, resource in resources.items():
            combined_text = f"{resource.get('name', '')} {resource.get('description', '')} {resource.get('type', '')}"
            tokens = tokenize(combined_text)
            documents.append(tokens)
            metadata.append((resource, user_id, res_id))

    # Build vocabulary
    vocab = sorted(set(token for doc in documents for token in doc))
    vocab_index = {word: idx for idx, word in enumerate(vocab)}

    # Compute TF vectors for resources
    tf_vectors = [compute_tf_vector(doc, vocab_index) for doc in documents]

    # Compute TF vector for query
    query_tokens = tokenize(query)
    query_vec = compute_tf_vector(query_tokens, vocab_index)

    # Compute cosine similarities
    similarities = []
    for i, tf_vector in enumerate(tf_vectors):
        score = cosine_similarity(query_vec, tf_vector)
        similarities.append((i, score))

    # Sort by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:top_n]

    # Construct result list
    results = []
    for idx, score in top_matches:
        res_info, user_id, res_id = metadata[idx]
        result = {
            "name": res_info.get("name", ""),
            "description": res_info.get("description", ""),
            "type": res_info.get("type", ""),
            "availability": res_info.get("availability", "unknown"),  # Ensure availability is included
            "score": round(score, 4),
            "user_id": user_id,
            "resource_id": res_id
        }
        results.append(result)

    return results



def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t]

def compute_tf_vector(tokens, vocab_index):
    vector = [0] * len(vocab_index)
    for token in tokens:
        if token in vocab_index:
            vector[vocab_index[token]] += 1
    return vector

def cosine_similarity(vec1, vec2):
    dot = sum(x * y for x, y in zip(vec1, vec2))
    norm1 = sqrt(sum(x * x for x in vec1))
    norm2 = sqrt(sum(y * y for y in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


