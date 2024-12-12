from transformers import WhisperForConditionalGeneration, WhisperTokenizer 
import torch
from datasets import Dataset, Audio

def load_model():
    model = WhisperForConditionalGeneration.from_pretrained('./model')
    model_tokenizer = WhisperTokenizer.from_pretrained('./model')

    print("Model loaded!")

    return model, model_tokenizer 
