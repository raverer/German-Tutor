# ===========================================
# Deutsch–Nepali–English AI Tutor + OCR Translator
# by Rajib Rawal
# ===========================================
# Models used:
#   - Helsinki-NLP/opus-mt-de-en  (German → English)
#   - Helsinki-NLP/opus-mt-en-de  (English → German)
#   - Hemg/english-To-Nepali-TRanslate  (English → Nepali)
#   - iamTangsang/MarianMT-Nepali-to-English  (Nepali → English)
#   - EasyOCR for image text extraction (English, German, Nepali)
# ===========================================

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gtts import gTTS
import tempfile
import torch
import warnings
from huggingface_hub import login
from PIL import Image
import numpy as np
import easyocr

# --------------------------------
# Setup
# --------------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Deutsch–Nepali Tutor", page_icon="🗣️", layout="centered")

st.title("🗣️ Deutsch–Nepali–English AI Tutor")
st.write("🎧 Speak, type, or upload an image — I'll translate and even speak it back!")

# --------------------------------
# Hugging Face Login
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

    # --- Speech Model ---
    whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    # --- OCR Model ---
    ocr_reader = easyocr.Reader(['en', 'de', 'ne'], gpu=False)

    return (
        tok_de_en, mod_de_en,
        tok_en_de, mod_en_de,
        tok_en_ne, mod_en_ne,
        tok_ne_en, mod_ne_en,
        whisper_asr, ocr_reader
    )

(
    tok_de_en, mod_de_en,
    tok_en_de, mod_en_de,
    tok_en_ne, mod_en_ne,
    tok_ne_en, mod_ne_en,
    whisper_asr, ocr_reader
) = load_models()

st.sidebar.success("✅ All models loaded successfully!")

# --------------------------------
# Translation Function (unchanged)
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

    # English → Nepali
    elif source == "English" and target == "Nepali":
        inputs = tok_en_ne(text, return_tensors="pt", padding=True)
        outputs = mod_en_ne.generate(**inputs, max_new_tokens=80)
        return tok_en_ne.decode(outputs[0], skip_special_tokens=True)

    # Nepali → English
    elif source == "Nepali" and target == "English":
        inputs = tok_ne_en(text, return_tensors="pt", padding=True)
        outputs = mod_ne_en.generate(**inputs, max_new_tokens=80)
        return tok_ne_en.decode(outputs[0], skip_special_tokens=True)

    # German → Nepali (via English)
    elif source == "German" and target == "Nepali":
        english = translate_text(text, "German", "English")
        return translate_text(english, "English", "Nepali")

    # Nepali → German (via English)
    elif source == "Nepali" and target == "German":
        english = translate_text(text, "Nepali", "English")
        return translate_text(english, "English", "German")

    else:
        return "⚠️ Unsupported translation direction."

# --------------------------------
# Mode Selector
# --------------------------------
mode = st.sidebar.radio("Choose Mode:", ["💬 Text/Speech Translator", "🖼️ Image Translator"])

# --------------------------------
# TEXT / SPEECH TRANSLATOR
# --------------------------------
if mode == "💬 Text/Speech Translator":
    source_lang = st.selectbox("🎙️ Source Language", ["German", "English", "Nepali"])
    target_lang = st.selectbox("🗣️ Target Language", ["German", "English", "Nepali"])
    input_mode = st.radio("Input Mode", ["⌨️ Type", "🎤 Speak"])

    text_input = ""

    if input_mode == "🎤 Speak":
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

                # Speech Output
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

# --------------------------------
# IMAGE TRANSLATOR
# --------------------------------
elif mode == "🖼️ Image Translator":
    st.subheader("🖼️ Upload an Image to Extract & Translate Text")
    image_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if image_file:
        try:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("🔍 Extracting text from image..."):
                results = ocr_reader.readtext(np.array(image))
                extracted_text = " ".join([res[1] for res in results])

            if extracted_text.strip():
                st.success(f"📝 Extracted Text:\n\n{extracted_text}")
                target_lang = st.selectbox("🌍 Translate to:", ["German", "English", "Nepali"])
                if st.button("Translate Image Text"):
                    translated = translate_text(extracted_text, "English", target_lang)
                    st.success(f"✅ Translated Text ({target_lang}):")
                    st.text_area("", translated, height=100)
            else:
                st.warning("No readable text found in the image.")

        except Exception as e:
            st.error(f"⚠️ OCR or translation failed: {e}")

st.markdown("---")
st.caption("Built with ❤️ by Rajib Rawal using Streamlit, Hugging Face, Whisper & EasyOCR")
