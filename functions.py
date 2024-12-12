from transformer import WhisperForConditionalGeneration, WhisperTokenizer 

def load_model():
    model = WhisperForConditionalGeneration.from_pretrained()