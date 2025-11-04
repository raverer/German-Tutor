# ===========================================
# Deutsch–Nepali–English AI Tutor + Image OCR
# by Rajib Rawal
# ===========================================
# Features:
#   1️⃣ Type-to-Translate
#   2️⃣ Speak-to-Translate (Whisper)
#   3️⃣ Image-to-Translate (OCR)
# ===========================================

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gtts import gTTS
import easyocr
import tempfile
import numpy as np
from PIL import Image
import torch
import warnings
from huggingface_hub import login

# --------------------------------
# Setup
# --------------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Deutsch–Nepali Tutor", page_icon="🗣️", layout="centered")

st.title("🗣️ Deutsch–Nepali–English AI Tutor")
st.write("🎧 Type, speak, or upload an image — I’ll translate and speak it back!")

# --------------------------------
# Login to Hugging Face (token stored securely in secrets)
# --------------------------------
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    login(token=HF_TOKEN)
    st.sidebar.success("🔐 Logged in to Hugging Face successfully!")
except Exception as e:
    st.sidebar.warning(f"⚠️ Hugging Face login skipped or failed: {e}")

# --------------------------------
# Model Loader
# --------------------------------
@st.cache_resource
def load_models():
    # --- Translation Models ---
    model_name_de_en = "Helsinki-NLP/opus-mt-de-en"
    model_name_en_de = "Helsinki-NLP/opus-mt-en-de"
    model_name_en_ne = "Hemg/english-To-Nepali-TRanslate"
    model_name_ne_en = "iamTangsang/MarianMT-Nepali-to-English"

    tok_de_en = AutoTokenizer.from_pretrained(model_name_de_en)
    mod_de_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_de_en)

    tok_en_de = AutoTokenizer.from_pretrained(model_name_en_de)
    mod_en_de = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_de)

    tok_en_ne = AutoTokenizer.from_pretrained(model_name_en_ne)
    mod_en_ne = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_ne)

    tok_ne_en = AutoTokenizer.from_pretrained(model_name_ne_en)
    mod_ne_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_ne_en)

    # --- Whisper Speech-to-Text ---
    whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    # --- OCR (Image-to-Text) ---
    ocr_reader_en_ne = easyocr.Reader(['en', 'ne'])
    ocr_reader_en_de = easyocr.Reader(['en', 'de'])

    return (
        tok_de_en, mod_de_en,
        tok_en_de, mod_en_de,
        tok_en_ne, mod_en_ne,
        tok_ne_en, mod_ne_en,
        whisper_asr,
        ocr_reader_en_ne, ocr_reader_en_de
    )

# Load all models
(
    tok_de_en, mod_de_en,
    tok_en_de, mod_en_de,
    tok_en_ne, mod_en_ne,
    tok_ne_en, mod_ne_en,
    whisper_asr,
    ocr_reader_en_ne, ocr_reader_en_de
) = load_models()

st.sidebar.success("✅ All models loaded successfully!")

# --------------------------------
# Translation Function
# --------------------------------
def translate_text(text, source, target):
    text = text.strip()
    if not text:
        return "⚠️ No text provided."

    # --- Direct Translations ---
    if source == "German" and target == "English":
        inputs = tok_de_en(text, return_tensors="pt", padding=True)
        outputs = mod_de_en.generate(**inputs)
        return tok_de_en.decode(outputs[0], skip_special_tokens=True)

    elif source == "English" and target == "German":
        inputs = tok_en_de(text, return_tensors="pt", padding=True)
        outputs = mod_en_de.generate(**inputs)
        return tok_en_de.decode(outputs[0], skip_special_tokens=True)

    elif source == "English" and target == "Nepali":
        inputs = tok_en_ne(text, return_tensors="pt", padding=True)
        outputs = mod_en_ne.generate(**inputs, max_new_tokens=80)
        return tok_en_ne.decode(outputs[0], skip_special_tokens=True)

    elif source == "Nepali" and target == "English":
        inputs = tok_ne_en(text, return_tensors="pt", padding=True)
        outputs = mod_ne_en.generate(**inputs, max_new_tokens=80)
        return tok_ne_en.decode(outputs[0], skip_special_tokens=True)

    # --- Cross-Language via English ---
    elif source == "German" and target == "Nepali":
        english = translate_text(text, "German", "English")
        return translate_text(english, "English", "Nepali")

    elif source == "Nepali" and target == "German":
        english = translate_text(text, "Nepali", "English")
        return translate_text(english, "English", "German")

    else:
        return "⚠️ Unsupported translation direction."


# --------------------------------
# Streamlit Interface
# --------------------------------
source_lang = st.selectbox("🎙️ Source Language", ["German", "English", "Nepali"])
target_lang = st.selectbox("🗣️ Target Language", ["German", "English", "Nepali"])
mode = st.radio("Input Mode", ["⌨️ Type", "🎤 Speak", "🖼️ Image"])

text_input = ""

# --- 1️⃣ TYPE MODE ---
if mode == "⌨️ Type":
    text_input = st.text_area("Enter text:", height=100)

# --- 2️⃣ SPEAK MODE ---
elif mode == "🎤 Speak":
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

# --- 3️⃣ IMAGE MODE ---
elif mode == "🖼️ Image":
    image_file = st.file_uploader("📷 Upload image", type=["png", "jpg", "jpeg"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Extracting text from image..."):
            if source_lang in ["English", "Nepali"]:
                results = ocr_reader_en_ne.readtext(np.array(image))
            else:
                results = ocr_reader_en_de.readtext(np.array(image))

            extracted_text = " ".join([res[1] for res in results])
        st.text_area("📝 Extracted Text", extracted_text, height=100)
        text_input = extracted_text

# --- TRANSLATE BUTTON ---
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
        st.warning("Please enter, speak, or upload some text first.")

st.markdown("---")
st.caption("Built with ❤️ by Rajib Rawal using Streamlit, Hugging Face, Whisper & EasyOCR")
