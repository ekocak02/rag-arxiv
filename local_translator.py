import logging
from transformers import pipeline
import torch
import nltk.data
import nltk

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Plan A2 - Translation Model (Helsinki-NLP)
# (Using the 'opus-mt-tc-big-en-tr' model)
MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-tr"

# --- Global Variables (for Caching/Singleton) ---
TRANSLATOR_PIPELINE = None
NLTK_PUNKT_LOADED = False
# ---

def initialize_nltk():
    """
    Downloads the NLTK 'punkt' model (sentence tokenizer) if not already present.
    'nltk.download' only runs on the first call.
    """
    global NLTK_PUNKT_LOADED
    if NLTK_PUNKT_LOADED:
        return
    try:
        # Check if the model is already loaded
        nltk.data.find('tokenizers/punkt')
        logging.info("NLTK 'punkt' model is already loaded.")
    except LookupError:
        # If the model is not loaded, download it
        logging.info("Downloading NLTK 'punkt' model (for sentence tokenization)...")
        # 'quiet=True' suppresses download logs
        nltk.download('punkt', quiet=True) 
        logging.info("✅ NLTK 'punkt' model downloaded.")
    NLTK_PUNKT_LOADED = True


def initialize_translator():
    """
    Loads the Helsinki-NLP model (pipeline) only once (lazy loading).
    Ensures the model is cached in the global 'TRANSLATOR_PIPELINE' variable.
    """
    global TRANSLATOR_PIPELINE
    if TRANSLATOR_PIPELINE is not None:
        return # Model is already loaded

    try:
        logging.info(f"Loading local translation model ({MODEL_NAME}) for the first time...")
        
        # Check for GPU (CUDA or Apple MPS)
        device = "cuda:0" if torch.cuda.is_available() else \
                 ("mps" if torch.backends.mps.is_available() else "cpu")
        
        if device != "cpu":
            logging.info(f"Device set to GPU: {device}")
        else:
            logging.info("Device set to CPU (translation may be slow).")

        TRANSLATOR_PIPELINE = pipeline(
            "translation_en_to_tr",
            model=MODEL_NAME,
            device=device
        )
        logging.info(f"✅ Local translation model loaded successfully (Device: {device}).")
        
    except Exception as e:
        logging.error(f"❌ Critical error while loading translation model: {e}")
        TRANSLATOR_PIPELINE = None


def translate_to_turkish(english_text: str) -> str:
    """
    Translates English text to Turkish using an advanced splitting strategy.
    
    Strategy:
    1. Splits text into paragraphs (by '\n') to preserve formatting (e.g., markdown).
    2. Splits each paragraph into sentences (using NLTK) for accurate translation.
    3. Reassembles the translated parts.
    """
    global TRANSLATOR_PIPELINE
    
    try:
        # 1. Ensure required models (Translator and NLTK) are loaded
        initialize_translator()
        initialize_nltk() 
        
        if TRANSLATOR_PIPELINE is None:
            raise Exception("Translation model could not be loaded.")

        logging.info("Translation process starting (Advanced Sentence Splitting)...")
        
        # 2. Hybrid Splitting Strategy
        
        # First, split the text into paragraphs (by newlines)
        paragraphs = english_text.split('\n')
        translated_paragraphs = []

        for paragraph in paragraphs:
            # Trim whitespace from the paragraph
            clean_paragraph = paragraph.strip()
            
            if not clean_paragraph:
                # This was an empty line (paragraph break), preserve the format
                translated_paragraphs.append("")
                continue

            # 3. Split the non-empty paragraph into sentences using 'nltk'
            # e.g.: "**Theme 3** This is a sentence." -> ["**Theme 3**", "This is a sentence."]
            sentences = nltk.sent_tokenize(clean_paragraph)
            
            # 4. Batch-translate the sentences (faster on GPU)
            # Note: If 'sentences' is empty, pipeline returns an empty list.
            translated_chunks = TRANSLATOR_PIPELINE(sentences)
            
            # 5. Rejoin the translated sentences (chunks) back into a paragraph
            translated_paragraph_parts = []
            for chunk in translated_chunks:
                translated_paragraph_parts.append(chunk['translation_text'])
            
            # Rebuild the paragraph by joining sentences with a space
            translated_paragraphs.append(' '.join(translated_paragraph_parts))

        # 6. Join all translated paragraphs with a newline ('\n')
        final_report = '\n'.join(translated_paragraphs)
        
        logging.info("✅ Translation process complete.")
        return final_report

    except Exception as e:
        logging.error(f"❌ Critical error during local translation: {e}")
        return f"[TRANSLATION FAILED: {e}]"


if __name__ == "__main__":
    """
    Updated test block to verify the advanced splitting strategy,
    especially with markdown formatting.
    """
    logging.info("Local Translator Module (local_translator.py) testing...")
    
    # Test text including Markdown headers and multi-sentence paragraphs
    test_text = """
**Theme 3: Understanding Challenges in LLM Reasoning**

This is the first sentence of the theme. This is the second sentence, which explains the first.

**Theme 4: Data Management**
This is a single sentence for Theme 4.
"""
    
    print("\n------------------------------ Original Text (English) ------------------------------")
    print(test_text)
    
    translated_text = translate_to_turkish(test_text)
    
    print("\n------------------------------ Translated Text (Turkish) ------------------------------")
    print(translated_text)

    logging.info("\nTest complete. The 'Theme 3' and 'Theme 4' headers should also be translated.")