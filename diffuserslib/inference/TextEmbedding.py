import torch
import re
import os
from typing import Dict
from ..FileUtils import getPathsFiles
from ..StringUtils import findBetween


def getClassFromFilename(path):
    filename = os.path.basename(path)
    index = filename.find("_")
    if index == -1:
        return None
    else:
        return filename[:index]


class TextEmbedding:
    def __init__(self, embedding_vectors, token: str, embedclass: str = None):
        self.embedding_vectors = embedding_vectors
        self.token = token
        self.embedclass = embedclass

    @classmethod
    def from_file(cls, embedding_path, token = None):
        if(token is None):
            token = findBetween(embedding_path, '<', '>', True)
        embedclass = getClassFromFilename(embedding_path)
        learned_embeds = torch.load(embedding_path, map_location="cpu")
        if ('string_to_param' in learned_embeds):  # .pt embedding
            string_to_token = learned_embeds['string_to_token']
            trained_token = list(string_to_token.keys())[0]
            if(token is None):
                token = trained_token
            string_to_param = learned_embeds['string_to_param']
            embedding_vectors = string_to_param[trained_token]
        else: # .bin diffusers concept
            trained_token = list(learned_embeds.keys())[0]
            if(token is None):
                token = trained_token
            embedding_vector = learned_embeds[trained_token]
            if (embedding_vector.ndim == 1):
                embedding_vectors = [embedding_vector]
            else:
                embedding_vectors = embedding_vector
        return cls(embedding_vectors, token, embedclass)


    def add_to_model(self, text_encoder, tokenizer):
        print(f"adding embedding token {self.token}")
        dtype = text_encoder.get_input_embeddings().weight.dtype
        for i, embedding_vector in enumerate(self.embedding_vectors):
            tokenpart = self.token + str(i)
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
        self.embeddings: Dict[str, TextEmbedding] = {} # Map of token to embedding
        self.modifiers: Dict[str, list[str]] = {} # dictionary of prompt modifiers

    def load_directory(self, path: str, base: str):
        print(f'Loading text embeddings for base {base} from path {path}')
        for embedding_path, embedding_file in getPathsFiles(f"{path}/*"):
            if (embedding_file.endswith('.bin') or embedding_file.endswith('.pt')):
                self.load_file(embedding_path)

    def load_file(self, path: str, token: str = None):
        embedding = TextEmbedding.from_file(path, token)
        self.embeddings[embedding.token] = embedding
        if(embedding.embedclass not in self.modifiers):
            self.modifiers[embedding.embedclass] = []
        self.modifiers[embedding.embedclass].append(embedding.token)
        print(f"Loaded embedding token {embedding.token} from file {path} with {len(embedding.embedding_vectors)} vectors")
        return embedding

    def add_to_model(self, text_encoder, tokenizer):
        for embedding in self.embeddings.values():
            embedding.add_to_model(text_encoder, tokenizer)

    def process_prompt(self, prompt: str):
        """ Expand token between angle brackets in prompt to a token for each vector in the embedding
            Use all vectors: <token>
            Use specific vectors: <token[0][4]>
        """
        prompttokens = re.findall(r'<.*?>', prompt)
        for prompttoken in prompttokens:
            tokenname = re.sub(r'\[[^\]]*\]', '', prompttoken) # remove everything between square brackets
            options = re.findall(r'\[(.*?)\]', prompttoken) # get everything between square brackets
            if (tokenname in self.embeddings):
                embedding = self.embeddings[tokenname]
                expandedtoken = ''
                if(len(options) == 0):
                    # use all vectors in token
                    for i, _ in enumerate(embedding.embedding_vectors):
                        expandedtoken = expandedtoken + ' ' + embedding.token + str(i)
                else:
                    # use selected vectors
                    for option in options:
                        expandedtoken = expandedtoken + ' ' + embedding.token + option
                prompt = prompt.replace(prompttoken, expandedtoken)
        return prompt
