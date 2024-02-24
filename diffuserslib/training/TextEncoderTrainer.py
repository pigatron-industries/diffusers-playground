from typing import List
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.logging import get_logger

logger = get_logger(__name__)

class TextEncoderTrainer():

    def __init__(self, tokenizer:CLIPTokenizer, text_encoder:CLIPTextModel, accelerator):
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.placeholder_tokens:List[str] = []
        self.placeholder_token_ids:List[int] = []


    def get_token_count(self, prompt:str):
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    
    def add_tokens(self, placeholder_tokens:List[str], initializer_tokens:str):
        self.placeholder_tokens = placeholder_tokens
        num_added_tokens = self.tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != len(placeholder_tokens):
            raise ValueError(f"The tokenizer already contains the token {placeholder_tokens}. Please pass a different `placeholder_token` that is not already in the tokenizer."
            )
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)

        self.initializer_token_ids = self.tokenizer.encode(initializer_tokens, add_special_tokens=False)
        print(f"initializer phrase has {len(self.initializer_token_ids)} tokens")
        
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for i, placeholder_token_id in enumerate(self.placeholder_token_ids):
                if i < len(self.initializer_token_ids):
                    token_embeds[placeholder_token_id] = token_embeds[self.initializer_token_ids[i]].clone()
                else:
                    token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[0])


    def store_original_embeddings(self):
        # keep original embeddings as reference
        self.orig_embeds = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()


    def restore_original_embeddings(self):
        """Restore the original embeddings except for placeholder tokens"""
        index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
        index_no_updates[min(self.placeholder_token_ids) : max(self.placeholder_token_ids) + 1] = False
        with torch.no_grad():
            self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[index_no_updates] = self.orig_embeds[index_no_updates]

    
    def get_learned_embeds(self):
        """Return the embeddings of the placeholder tokens"""
        return self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[min(self.placeholder_token_ids) : max(self.placeholder_token_ids) + 1].detach().cpu()


    def fetch_text_encoder_parameters(self):
        self.text_lora_parameters_one = list(filter(lambda p: p.requires_grad, self.text_encoder.parameters()))            
