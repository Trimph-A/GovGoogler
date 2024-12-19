import re

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()
    return text
