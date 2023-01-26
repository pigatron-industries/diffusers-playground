

class TextEmbedding:
    def __init__(self, embedding, token: str, expandedtoken: str = None):
        self.embedding = embedding
        self.token = token
        self.expandedtoken = expandedtoken


class TextEmbeddings:
    def __init__(self, base: str, embeddings: list[TextEmbedding] = []):
        self.base = base
        self.embeddings = embeddings

