# app.py
import streamlit as st
import torch, gc, io, librosa, numpy as np
from PIL import Image
import easyocr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------------------------------
# Streamlit Config
# -------------------------------------------------------
st.set_page_config(page_title="Multimodal Translator (Final)", layout="centered")
st.title("🎧🖼️ Multimodal Translator — Speech · OCR · Text")
st.caption("Whisper-Base (chunked) + EasyOCR + DE↔EN + EN↔NE + NE→EN (iamTangsang) — Optimized for Streamlit Cloud")

# -------------------------------------------------------
# Model IDs
# -------------------------------------------------------
MODEL_WHISPER = "openai/whisper-base"
MODEL_DE_TO_EN = "Helsinki-NLP/opus-mt-de-en"
MODEL_EN_TO_DE = "Helsinki-NLP/opus-mt-en-de"
MODEL_EN_TO_NE = "Hemg/english-To-Nepali-TRanslate"
MODEL_NE_TO_EN = "iamTangsang/MarianMT-Nepali-to-English"

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def resize_image(uploaded_file, max_px=1280, jpeg_quality=85):
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf)

# -------------------------------------------------------
# Cached Model Loaders (lazy loading)
# -------------------------------------------------------
@st.cache_resource
def get_whisper():
    return pipeline("automatic-speech-recognition", model=MODEL_WHISPER)

@st.cache_resource
def get_ocr():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def get_de_en():
    tok = AutoTokenizer.from_pretrained(MODEL_DE_TO_EN)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DE_TO_EN)
    return model, tok

@st.cache_resource
def get_en_de():
    tok = AutoTokenizer.from_pretrained(MODEL_EN_TO_DE)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_EN_TO_DE)
    return model, tok

@st.cache_resource
def get_en_ne():
    tok = AutoTokenizer.from_pretrained(MODEL_EN_TO_NE)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_EN_TO_NE)
    return model, tok

@st.cache_resource
def get_ne_en():
    tok = AutoTokenizer.from_pretrained(MODEL_NE_TO_EN)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NE_TO_EN)
    return model, tok

# -------------------------------------------------------
# Whisper Chunked Transcription
# -------------------------------------------------------
def transcribe_long_audio_file(path, chunk_length_s=30, overlap_s=2):
    asr = get_whisper()
    audio, sr = librosa.load(path, sr=16000)
    chunk_samples = int(chunk_length_s * sr)
    overlap_samples = int(overlap_s * sr)
    texts = []
    for start in range(0, len(audio), chunk_samples - overlap_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        chunk_input = {"array": chunk, "sampling_rate": sr}
        res = asr(chunk_input)
        texts.append(res.get("text", ""))
        if end == len(audio):
            break
    clear_memory()
    return " ".join(texts)

# -------------------------------------------------------
# Translation Helper
# -------------------------------------------------------
def translate_with_model(model_tok_pair, text, max_new_tokens=200):
    model, tokenizer = model_tok_pair
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# -------------------------------------------------------
# Translation Pipelines
# -------------------------------------------------------
def de_to_en(t): return translate_with_model(get_de_en(), t)
def en_to_de(t): return translate_with_model(get_en_de(), t)
def en_to_ne(t): return translate_with_model(get_en_ne(), t)
def ne_to_en(t): return translate_with_model(get_ne_en(), t)

def de_to_ne(t):
    en = de_to_en(t)
    return en_to_ne(en)

def ne_to_de(t):
    en = ne_to_en(t)
    return en_to_de(en)

# -------------------------------------------------------
# Streamlit Interface
# -------------------------------------------------------
mode = st.radio("Select Mode:", [
    "🎤 Audio → Text (Whisper-Base)",
    "🖼️ Image → Text (EasyOCR)",
    "🔤 Text Translation (DE↔EN↔NE)"
])

# ---------------- AUDIO ----------------
if mode.startswith("🎤"):
    st.subheader("Upload Audio File (mp3 / wav / m4a)")
    audio_file = st.file_uploader("Audio", type=["mp3", "wav", "m4a"])
    if audio_file:
        st.audio(audio_file)
        tmp_path = "temp_audio.wav"
        with open(tmp_path, "wb") as f:
            f.write(audio_file.read())
        with st.spinner("Transcribing with Whisper-Base (chunked)..."):
            text = transcribe_long_audio_file(tmp_path)
            st.success("✅ Transcription Complete")
            st.text_area("Transcribed Text", value=text, height=250)

# ---------------- OCR ----------------
elif mode.startswith("🖼️"):
    st.subheader("Upload Image for OCR")
    img_file = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = resize_image(img_file)
        st.image(img, use_container_width=True, caption="Resized image (<1 MB)")
        with st.spinner("Extracting text with EasyOCR..."):
            reader = get_ocr()
            txt = "\n".join(reader.readtext(np.array(img), detail=0))
            st.success("✅ OCR Complete")
            st.text_area("Extracted Text", value=txt, height=250)
        clear_memory()

# ---------------- TRANSLATION ----------------
else:
    st.subheader("Text Translation")
    direction = st.selectbox("Select Translation Direction", [
        "German → English",
        "English → German",
        "English → Nepali",
        "Nepali → English",
        "German → Nepali (via English)",
        "Nepali → German (via English)"
    ])
    text_in = st.text_area("Enter text here:", height=200)

    if st.button("Translate"):
        if not text_in.strip():
            st.warning("Please enter text to translate.")
        else:
            with st.spinner("Translating..."):
                try:
                    if direction == "German → English":
                        out = de_to_en(text_in)
                    elif direction == "English → German":
                        out = en_to_de(text_in)
                    elif direction == "English → Nepali":
                        out = en_to_ne(text_in)
                    elif direction == "Nepali → English":
                        out = ne_to_en(text_in)
                    elif direction == "German → Nepali (via English)":
                        out = de_to_ne(text_in)
                    elif direction == "Nepali → German (via English)":
                        out = ne_to_de(text_in)
                    st.success("✅ Translation Complete")
                    st.text_area("Translated Text:", value=out, height=250)
                except Exception as e:
                    st.error(f"Translation failed: {e}")
            clear_memory()

st.markdown("---")
st.caption("Designed for Nepali students.")
