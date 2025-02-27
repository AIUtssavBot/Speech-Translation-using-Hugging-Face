# Import necessary modules
import streamlit as st
from vosk import Model, KaldiRecognizer
import pyaudio
from transformers import pipeline
import pyttsx3
import json

# Load Vosk model for offline Speech-to-Text (S2T)
def load_vosk_model():
    try:
        # Replace with the absolute path to your model directory
        model_path = "C:/Users/sigma/ml projects/GenAI project/vosk-model-small-en-us-0.15"
        model = Model(model_path)
        recognizer = KaldiRecognizer(model, 16000)
        return recognizer
    except Exception as e:
        st.error(f"Error loading Vosk model: {e}")
        return None


# Speech-to-Text (S2T) Function using Vosk
def speech_to_text(recognizer):
    if not recognizer:
        return "Recognizer not loaded."
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    stream.start_stream()
    st.write("Listening... Speak now")

    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            stream.stop_stream()
            stream.close()
            audio.terminate()
            return result['text']

# Text-to-Text (T2T) Translation using Hugging Face Transformers
# def translate_text(text, model_name):
#     translator = pipeline("translation", model=model_name)
#     translated = translator(text)[0]['translation_text']
#     return translated

# Text-to-Speech (T2S) Function using pyttsx3
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit Interface for Multi-Speaker Translator
# def main():
#     st.title("Real-Time Multi-Speaker Voice Translator")

#     # Load Vosk model
#     recognizer = load_vosk_model()
#     if not recognizer:
#         st.error("Failed to initialize the speech recognizer. Please ensure the Vosk model is correctly installed and the path is correct.")
#         return

#     # Language selection
#     lang_a = st.selectbox("Select User A's Language", ["en", "fr", "es"])
#     lang_b = st.selectbox("Select User B's Language", ["en", "fr", "es"])

#     # Define language models for translation
#     models = {
#         ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
#         ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
#         ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
#         ("es", "en"): "Helsinki-NLP/opus-mt-es-en"
#     }

#     if st.button("Start Conversation"):
#         # User A speaks
#         st.write("User A: Please speak...")
#         text_a = speech_to_text(recognizer)
#         st.write(f"User A said: {text_a}")

#         # Translate to User B's language
#         model_ab = models.get((lang_a, lang_b))
#         translated_a = translate_text(text_a, model_ab)
#         st.write(f"Translation for User B: {translated_a}")
#         text_to_speech(translated_a)

#         # User B speaks
#         st.write("User B: Please speak...")
#         text_b = speech_to_text(recognizer)
#         st.write(f"User B said: {text_b}")

#         # Translate to User A's language
#         model_ba = models.get((lang_b, lang_a))
#         translated_b = translate_text(text_b, model_ba)
#         st.write(f"Translation for User A: {translated_b}")
#         text_to_speech(translated_b)

# if __name__ == "__main__":
#     main()

def translate_text(text, model_name):
    if not model_name:
        st.error("Translation model not found for the selected language pair.")
        return "Translation model not available"
    try:
        translator = pipeline("translation", model=model_name)
        translated = translator(text)[0]['translation_text']
        return translated
    except Exception as e:
        st.error(f"Error in translation: {e}")
        return "Translation failed."

def main():
    st.title("Real-Time Multi-Speaker Voice Translator")

    recognizer = load_vosk_model()
    if not recognizer:
        st.error("Failed to initialize the speech recognizer.")
        return

    # Language selection
    lang_a = st.selectbox("Select User A's Language", ["en", "fr", "es"])
    lang_b = st.selectbox("Select User B's Language", ["en", "fr", "es"])

    # Define translation models
    models = {
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
        ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
        ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
        # Add more language models as needed
    }

    if st.button("Start Conversation"):
        # User A speaks
        st.write("User A: Please speak...")
        text_a = speech_to_text(recognizer)
        st.write(f"User A said: {text_a}")

        # Translation for User B
        model_ab = models.get((lang_a, lang_b))
        translated_a = translate_text(text_a, model_ab)
        st.write(f"Translation for User B: {translated_a}")
        text_to_speech(translated_a)

        # User B speaks
        st.write("User B: Please speak...")
        text_b = speech_to_text(recognizer)
        st.write(f"User B said: {text_b}")

        # Translation for User A
        model_ba = models.get((lang_b, lang_a))
        translated_b = translate_text(text_b, model_ba)
        st.write(f"Translation for User A: {translated_b}")
        text_to_speech(translated_b)

if __name__ == "__main__":
    main()
