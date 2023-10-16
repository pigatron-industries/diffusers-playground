import torch
import re
import os
from typing import Dict, List
from .arch.StableDiffusionPipelines import DiffusersPipelineWrapper
from ..FileUtils import getPathsFiles
from ..StringUtils import findBetween

from safetensors import safe_open


def getClassFromFilename(path):
    filename = os.path.basename(path)
    index = filename.find("_")
    if index == -1:
        return None
    else:
        return filename[:index]


class TextEmbedding:
    def __init__(self, embeddings, token: str, embedclass: str = None, path: str = None):
        self.embeddings = embeddings
        self.token = token
        self.embedclass = embedclass
        self.path = path

    @classmethod
    def from_file(cls, embedding_path, token = None):
        if(token is None):
            token = findBetween(embedding_path, '<', '>', True)
        embedclass = getClassFromFilename(embedding_path)
        embeddings = []
        if(embedding_path.endswith('.safetensors')):
            with safe_open(embedding_path, framework='pt') as f:
                if ('clip_l' in f.keys()):
                    # Multiple embeddings with tokenizer as key
                    embeddings = []
                    embeddings.append(f.get_tensor('clip_l'))
                    embeddings.append(f.get_tensor('clip_g'))
                else:
                    # Single embedding with token as key
                    for key in f.keys():
                        if(token is None):
                            token = key
                        embeddings.append(f.get_tensor(key))
        else:
            learned_embeds = torch.load(embedding_path, map_location="cpu")
            if ('string_to_param' in learned_embeds):  
                # .pt embedding
                string_to_token = learned_embeds['string_to_token']
                trained_token = list(string_to_token.keys())[0]
                if(token is None):
                    token = trained_token
                string_to_param = learned_embeds['string_to_param']
                embedding_vectors = string_to_param[trained_token]
                embeddings.append(embedding_vectors)
            else: 
                # .bin diffusers concept
                trained_token = list(learned_embeds.keys())[0]
                if(token is None):
                    token = trained_token
                embedding_vector = learned_embeds[trained_token]
                if (embedding_vector.ndim == 1):
                    embeddings.append([embedding_vector])
                else:
                    embeddings.append(embedding_vector)
        return cls(embeddings, token, embedclass, embedding_path)


    def add_to_model(self, pipeline: DiffusersPipelineWrapper):
        print(f"adding embedding token {self.token}")
        pipeline.add_embeddings(self.token, self.embeddings)


class TextEmbeddings:
    def __init__(self, base: str):
        self.base: str = base
        self.embeddings: Dict[str, TextEmbedding] = {} # Map of token to embedding
        self.modifiers: Dict[str, list[str]] = {} # dictionary of prompt modifiers


    def load_directory(self, path: str, base: str):
        print(f'Loading text embeddings for base {base} from path {path}')
        for embedding_path, embedding_file in getPathsFiles(f"{path}/*"):
            if (embedding_file.endswith('.bin') or embedding_file.endswith('.pt') or embedding_file.endswith('.safetensors')):
                self.load_file(embedding_path)


    def load_file(self, path: str, token: str = None):
        embedding = TextEmbedding.from_file(path, token)
        self.embeddings[embedding.token] = embedding
        if(embedding.embedclass not in self.modifiers):
            self.modifiers[embedding.embedclass] = []
        self.modifiers[embedding.embedclass].append(embedding.token)
        print(f"Loaded embedding token {embedding.token} from file {path} with {len(embedding.embeddings[0])} vectors")
        return embedding


    def process_prompt_and_add_tokens(self, prompt: str, pipeline: DiffusersPipelineWrapper):
        tokens = self.get_tokens_from_prompt(prompt)
        self.add_tokens_to_model(pipeline, tokens)
        prompt = self.process_prompt(prompt)
        return prompt


    def add_tokens_to_model(self, pipeline: DiffusersPipelineWrapper, tokens: List[str]):
        for token in tokens:
            embedding = self.embeddings[token]
            try:
                embedding.add_to_model(pipeline)
            except ValueError:
                pass


    def get_tokens_from_prompt(self, prompt: str):
        prompttokens = re.findall(r'<.*?>', prompt)
        tokennames = []
        for prompttoken in prompttokens:
            tokenname = re.sub(r'\[[^\]]*\]', '', prompttoken)  # remove everything between square brackets
            tokennames.append(tokenname)
        return tokennames
    

    def process_prompt(self, prompt: str):
        """ Expand token between angle brackets in prompt to a token for each vector in the embedding
            Use all vectors: <token>
            Use specific vectors: <token[0][4]>
        """
        prompttokens = self.get_tokens_from_prompt(prompt)
        for prompttoken in prompttokens:
            tokenname = re.sub(r'\[[^\]]*\]', '', prompttoken) # remove everything between square brackets
            options = re.findall(r'\[(.*?)\]', prompttoken) # get everything between square brackets
            if (tokenname in self.embeddings):
                embedding = self.embeddings[tokenname]
                expandedtoken = ''
                if(len(options) == 0):
                    # use all vectors in token
                    for i in range(len(embedding.embeddings[0])):
                        expandedtoken = expandedtoken + ' ' + embedding.token + str(i)
                else:
                    # use selected vectors
                    for option in options:
                        expandedtoken = expandedtoken + ' ' + embedding.token + option
                prompt = prompt.replace(prompttoken, expandedtoken)
            else:
                print(f"WARNING: embedding token {tokenname} not found")
        return prompt
