import torch
from typing import Dict
from ..FileUtils import getPathsFiles
from ..StringUtils import findBetween


class TextEmbedding:
    def __init__(self, embedding_vectors, token: str, expandedtoken: str = None):
        self.embedding_vectors = embedding_vectors
        self.token = token
        self.expandedtoken = expandedtoken

    @classmethod
    def from_file(cls, embedding_path, token = None):
        if(token is None):
            token = findBetween(embedding_path, '<', '>', True)
        learned_embeds = torch.load(embedding_path, map_location="cpu")
        if ('string_to_param' in learned_embeds):  # .pt embedding
            string_to_token = learned_embeds['string_to_token']
            trained_token = list(string_to_token.keys())[0]
            if(token is None):
                token = trained_token
            string_to_param = learned_embeds['string_to_param']
            expandedtoken = ''
            embedding_vectors = string_to_param[trained_token]
            for i in range(len(embedding_vectors)):
                tokenpart = token + str(i)
                expandedtoken = expandedtoken + ' ' + tokenpart
            return cls(embedding_vectors, token, expandedtoken)
        else: # .bin diffusers concept
            trained_token = list(learned_embeds.keys())[0]
            if(token is None):
                token = trained_token
            embedding_vector = learned_embeds[trained_token]

            if (embedding_vector.ndim == 1):
                return cls([embedding_vector], token)
            else:
                return cls(embedding_vector, token)


    def add_to_model(self, text_encoder, tokenizer):
        print(f"adding embedding token {self.token}")
        dtype = text_encoder.get_input_embeddings().weight.dtype
        for i, embedding_vector in enumerate(self.embedding_vectors):
            tokenpart = self.token
            if(len(self.embedding_vectors) > 1):
                tokenpart = tokenpart + str(i)
            embedding_vector.to(dtype)
            num_added_tokens = tokenizer.add_tokens(tokenpart)
            if(num_added_tokens == 0):
                raise ValueError(f"The tokenizer already contains the token {tokenpart}")
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_id = tokenizer.convert_tokens_to_ids(tokenpart)
            text_encoder.get_input_embeddings().weight.data[token_id] = embedding_vector


class TextEmbeddings:
    def __init__(self, base: str):
        self.base: str = base
        self.embeddings: Dict[str, TextEmbedding] = {}

    def load_directory(self, path: str, base: str):
        print(f'Loading text embeddings for base {base} from path {path}')
        for embedding_path, embedding_file in getPathsFiles(f"{path}/*"):
            if (embedding_file.endswith('.bin') or embedding_file.endswith('.pt')):
                self.load_file(embedding_path, base)

    def load_file(self, path: str, base: str):
        embedding = TextEmbedding.from_file(path)
        self.embeddings[embedding.token] = embedding
        print(f"Loaded embedding token {embedding.token} from file {path} with {len(embedding.embedding_vectors)} vectors")

    def add_to_model(self, text_encoder, tokenizer):
        for embedding in self.embeddings.values():
            embedding.add_to_model(text_encoder, tokenizer)
