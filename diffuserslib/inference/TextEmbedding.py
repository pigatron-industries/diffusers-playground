import torch
import re
import os
from typing import Dict, List
from .arch.StableDiffusionPipelines import DiffusersPipelineWrapper
from ..FileUtils import getPathsFiles
from ..StringUtils import findBetween

from safetensors import safe_open
from numpy import ndarray


def getClassFromFilename(path):
    filename = os.path.basename(path)
    index = filename.find("_")
    if index == -1:
        return None
    else:
        return filename[:index]


class TextEmbedding:
    def __init__(self, embeddings:List[ndarray], token: str, embedclass: str|None = None, path: str|None = None):
        self.embeddings = embeddings
        self.token = token
        self.embedclass = embedclass
        self.path = path

    @classmethod
    def from_file(cls, embedding_path, token = None):
        if(token is None):
            token = findBetween(embedding_path, '<', '>', True)
        if(token is None):
            raise ValueError("Could not find token in filename")
        embedclass = getClassFromFilename(embedding_path)
        return cls([], token, embedclass, embedding_path)
    
    def loadFile(self):
        if(self.path is None):
            raise ValueError("No path set for embedding")
        print(f"Loading embedding token {self.token} from file {self.path}")
        self.embeddings = []
        if(self.path.endswith('.safetensors')):
            with safe_open(self.path, framework='pt') as f:
                if ('clip_l' in f.keys()):
                    # Multiple embeddings with tokenizer as key
                    self.embeddings = []
                    self.embeddings.append(f.get_tensor('clip_l'))
                    self.embeddings.append(f.get_tensor('clip_g'))
                else:
                    # Single embedding with token as key
                    for key in f.keys():
                        if(self.token is None):
                            self.token = key
                        self.embeddings.append(f.get_tensor(key))
        else:
            learned_embeds = torch.load(self.path, map_location="cpu")
            if ('string_to_param' in learned_embeds):  
                # .pt embedding
                string_to_token = learned_embeds['string_to_token']
                trained_token = list(string_to_token.keys())[0]
                if(self.token is None):
                    self.token = trained_token
                string_to_param = learned_embeds['string_to_param']
                embedding_vectors = string_to_param[trained_token]
                self.embeddings.append(embedding_vectors)
            elif ('clip_l' in learned_embeds):
                # Multiple embeddings with tokenizer as key
                self.embeddings = []
                self.embeddings.append(learned_embeds['clip_l'])
                self.embeddings.append(learned_embeds['clip_g'])
            else: 
                # .bin diffusers concept
                trained_token = list(learned_embeds.keys())[0]
                if(self.token is None):
                    self.token = trained_token
                embedding_vector = learned_embeds[trained_token]
                if (embedding_vector.ndim == 1):
                    self.embeddings.append([embedding_vector])
                else:
                    self.embeddings.append(embedding_vector)

    def getEmbedding(self):
        if(len(self.embeddings) == 0):
            self.loadFile()
        return self.embeddings

    def add_to_model(self, pipeline: DiffusersPipelineWrapper):
        print(f"adding embedding token {self.token}")
        pipeline.add_embeddings(self.token, self.getEmbedding())


class TextEmbeddings:
    def __init__(self, base: str):
        self.base: str = base
        self.embeddings: Dict[str, TextEmbedding] = {} # Map of token to embedding
        self.modifiers: Dict[str, list[str]] = {} # dictionary of prompt modifiers


    def load_directory(self, path: str, base: str):
        print(f'Loading text embeddings for base {base} from path {path}')
        for embedding_path, embedding_file in getPathsFiles(f"{path}/*") + getPathsFiles(f"{path}/**/*"):
            if (embedding_file.endswith('.bin') or embedding_file.endswith('.pt') or embedding_file.endswith('.safetensors')):
                try:
                    self.load_file(embedding_path)
                except ValueError as e:
                    print("WARNING: Could not load embedding file: " + embedding_path)
                    print(e)


    def load_file(self, path: str, token: str|None = None):
        embedding = TextEmbedding.from_file(path, token)
        self.embeddings[embedding.token] = embedding
        if(embedding.embedclass is not None):
            if(embedding.embedclass not in self.modifiers):
                self.modifiers[embedding.embedclass] = []
            self.modifiers[embedding.embedclass].append(embedding.token)
        # print(f"Loaded embedding token {embedding.token} from file {path} with {len(embedding.embeddings[0])} vectors")
        return embedding


    def process_prompt_and_add_tokens(self, prompt: str, pipeline: DiffusersPipelineWrapper):
        prompt, tokennames = self.process_prompt(prompt)
        self.add_tokens_to_model(pipeline, tokennames)
        return prompt


    def add_tokens_to_model(self, pipeline: DiffusersPipelineWrapper, tokens: List[str]):
        for token in tokens:
            embedding = self.embeddings[token]
            try:
                embedding.add_to_model(pipeline)
            except ValueError:
                pass


    def get_tokenstrings_from_prompt(self, prompt: str):
        return re.findall(r'<.*?>', prompt)
    
    def get_tokennames_from_prompt(self, prompt: str):
        prompttokens = self.get_tokenstrings_from_prompt(prompt)
        tokennames = []
        for prompttoken in prompttokens:
            tokenname = re.sub(r'\[[^\]]*\]', '', prompttoken) # remove everything between square brackets
            tokennames.append(tokenname)
        return tokennames
    

    def process_prompt(self, prompt: str):
        """ Expand token between angle brackets in prompt to a token for each vector in the embedding
            Use all vectors: <token>
            Use specific vectors: <token[0][4]>
            Randomize with a wildcard: <token*>
        """
        prompttokens = self.get_tokenstrings_from_prompt(prompt)
        tokennames = []
        for prompttoken in prompttokens:
            tokenname = re.sub(r'\[[^\]]*\]', '', prompttoken) # remove everything between square brackets
            options = re.findall(r'\[(.*?)\]', prompttoken) # get everything between square brackets
            if('*' in tokenname):
                tokenname = self.randomize_wildcard_token(tokenname)
            if (tokenname in self.embeddings):
                embedding = self.embeddings[tokenname]
                embedding.getEmbedding()
                expandedtoken = ''
                if(len(options) == 0):
                    # use all vectors in token
                    for i in range(len(embedding.embeddings[0])):
                        expandedtoken = expandedtoken + ' ' + embedding.token + str(i)
                else:
                    # use selected vectors
                    for option in options:
                        expandedtoken = expandedtoken + ' ' + embedding.token + option
                tokennames.append(tokenname)
                prompt = prompt.replace(prompttoken, expandedtoken)
            else:
                print(f"WARNING: embedding token {tokenname} not found")
        return prompt, tokennames


    def randomize_wildcard_token(self, tokenname: str) -> str:
        """ Replace wildcard * with random token from embedding """
        tokenregex = tokenname.replace('*', '.*')
        print(f"Randomizing wildcard token {tokenname} with regex {tokenregex}")
        matchingtokens = []
        for embedding in self.embeddings.values():
            if(re.match(tokenregex, embedding.token)):
                matchingtokens.append(embedding.token)
        if(len(matchingtokens) > 0):
            return matchingtokens[torch.randint(len(matchingtokens), (1,))]
        else:
            raise ValueError(f"Could not find any embedding token matching wildcard {tokenname}")