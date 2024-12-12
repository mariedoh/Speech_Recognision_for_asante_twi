from transformers import WhisperForConditionalGeneration, WhisperTokenizer 
import torch
from datasets import Dataset, Audio

def load_model():
    model = WhisperForConditionalGeneration.from_pretrained('./model')
    model_tokenizer = WhisperTokenizer.from_pretrained('./model')

    print("Model loaded!")

    return model, model_tokenizer 

def format_string(input_string):
    words = input_string.split()
    formatted_words = []

    for word in words:
        # Check if the word represents a number
        try:
            if '.' in word:  # Float detection
                number = float(word)
                formatted_words.append(f"{number:,}")
            else:  # Integer detection
                number = int(word)
                formatted_words.append(f"{number:,}")
        except ValueError:
            # If it's not a number, add the word as is
            formatted_words.append(word)

    return ' '.join(formatted_words)
