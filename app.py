# app.py
# Streamlit: Deutsch–Nepali–English Interactive Tutor (Colab + Streamlit compatible)
# Uses: Helsinki (De/En), Hemg (En->Ne & Ne->En - attempting with one model)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gtts import gTTS
import tempfile
import torch
from huggingface_hub import login

# Log in to Hugging Face Hub
try:
    login(token=userdata.get("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"))
except Exception as e:
    print(f"Could not log in to Hugging Face Hub: {e}")


# --------------------------------
# Page Setup
# --------------------------------
st.set_page_config(page_title="Deutsch–Nepali Tutor", page_icon="🗣️", layout="centered")
st.title("🗣️ Deutsch–Nepali–English AI Tutor")
st.write("🎧 Speak or type in German, English, or Nepali — I’ll translate and speak it back!")

# --------------------------------
# Model Loader
# --------------------------------
@st.cache_resource
def load_models():
    # German ↔ English
    model_name_de_en = "Helsinki-NLP/opus-mt-de-en"
    model_name_en_de = "Helsinki-NLP/opus-mt-en-de"

    tok_de_en = AutoTokenizer.from_pretrained(model_name_de_en)
    mod_de_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_de_en)

    tok_en_de = AutoTokenizer.from_pretrained(model_name_en_de)
    mod_en_de = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_de)

    # English ↔ Nepali (Using Hemg model for both directions)
    model_name_en_ne_ne_en = "Hemg/english-To-Nepali-TRanslate"
    tok_en_ne_ne_en = AutoTokenizer.from_pretrained(model_name_en_ne_ne_en)
    mod_en_ne_ne_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_ne_ne_en)

    # Whisper tiny for speech-to-text
    whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    return (tok_de_en, mod_de_en, tok_en_de, mod_en_de, tok_en_ne_ne_en, mod_en_ne_ne_en, whisper_asr)


# Update the return values to match the new load_models function
tok_de_en, mod_de_en, tok_en_de, mod_en_de, tok_en_ne_ne_en, mod_en_ne_ne_en, whisper_asr = load_models()

# --------------------------------
# Translation Function
# --------------------------------
def translate_text(text, source, target):
    text = text.strip()

    # German → English
    if source == "German" and target == "English":
        inputs = tok_de_en(text, return_tensors="pt", padding=True)
        outputs = mod_de_en.generate(**inputs)
        return tok_de_en.decode(outputs[0], skip_special_tokens=True)

    # English → German
    elif source == "English" and target == "German":
        inputs = tok_en_de(text, return_tensors="pt", padding=True)
        outputs = mod_en_de.generate(**inputs)
        return tok_en_de.decode(outputs[0], skip_special_tokens=True)

    # English → Nepali (Using Hemg model)
    elif source == "English" and target == "Nepali":
        inputs = tok_en_ne_ne_en(text, return_tensors="pt", padding=True)
        outputs = mod_en_ne_ne_en.generate(**inputs)
        return tok_en_ne_ne_en.decode(outputs[0], skip_special_tokens=True)

    # Nepali → English (Using Hemg model - may not be optimal)
    elif source == "Nepali" and target == "English":
        inputs = tok_en_ne_ne_en(text, return_tensors="pt", padding=True)
        outputs = mod_en_ne_ne_en.generate(**inputs)
        return tok_en_ne_ne_en.decode(outputs[0], skip_special_tokens=True)

    # Nepali → German (via English)
    elif source == "Nepali" and target == "German":
        english = translate_text(text, "Nepali", "English")
        return translate_text(english, "English", "German")

    # German → Nepali (via English)
    elif source == "German" and target == "Nepali":
        english = translate_text(text, "German", "English")
        return translate_text(english, "English", "Nepali")

    else:
        return "⚠️ Unsupported translation direction."


# --------------------------------
# Streamlit Interface
# --------------------------------
source_lang = st.selectbox("🎙️ Source Language", ["German", "English", "Nepali"])
target_lang = st.selectbox("🗣️ Target Language", ["German", "English", "Nepali"])
mode = st.radio("Input Mode", ["⌨️ Type", "🎤 Speak"])

text_input = ""

if mode == "🎤 Speak":
    st.write("🎧 Upload or record your voice (WAV/MP3)")
    audio_file = st.file_uploader("Upload file", type=["wav", "mp3"])

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        with st.spinner("Transcribing speech..."):
            result = whisper_asr(tmp_path)
            text_input = result["text"]
            st.success(f"🗒️ Transcribed: {text_input}")
else:
    text_input = st.text_area("Enter text:", height=100)

if st.button("Translate"):
    if text_input:
        with st.spinner("Translating..."):
            translated = translate_text(text_input, source_lang, target_lang)
            st.success("✅ Translation complete!")
            st.text_area("Translated Text:", translated, height=100)

            # --- Speech output ---
            try:
                tts_lang = (
                    "de" if target_lang == "German"
                    else "en" if target_lang == "English"
                    else "ne"
                )
                tts = gTTS(translated, lang=tts_lang)
                tts.save("output.mp3")
                st.audio("output.mp3", format="audio/mp3")
            except Exception:
                st.warning("Speech output not available for this language.")
    else:
        st.warning("Please enter or record some text first.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Hugging Face, and Whisper")
