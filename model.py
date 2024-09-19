from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    pause_token = "<pause>"
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'additional_special_tokens': [pause_token]})
    tokenizer.pause_token = pause_token
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    return model, tokenizer, pause_token
