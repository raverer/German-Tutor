import streamlit as st
import torch, gc, io, librosa, numpy as np, base64
from PIL import Image
import easyocr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS

# -------------------------------------------------------
# Streamlit Config
# -------------------------------------------------------
st.set_page_config(page_title="German Tutor(Final)", layout="centered")
st.title("German Translator — Audio · OCR · Text")

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

def text_to_speech(text, lang_code="en"):
    """Generate speech audio for translated text."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save("temp_tts.mp3")
        audio_file = open("temp_tts.mp3", "rb").read()
        b64 = base64.b64encode(audio_file).decode()
        md = f"""
        <audio controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        return md
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return None

# -------------------------------------------------------
# Cached Model Loaders
# -------------------------------------------------------
@st.cache_resource
def get_whisper(): return pipeline("automatic-speech-recognition", model=MODEL_WHISPER)
@st.cache_resource
def get_ocr(): return easyocr.Reader(['en'], gpu=False)
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
        res = asr(chunk_input, generate_kwargs={"language": "en"})
        texts.append(res.get("text", ""))
        if end == len(audio):
            break
    clear_memory()
    return " ".join(texts)

# -------------------------------------------------------
# Translation Helpers
# -------------------------------------------------------
def translate_with_model(model_tok_pair, text, max_new_tokens=200):
    model, tokenizer = model_tok_pair
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def de_to_en(t): return translate_with_model(get_de_en(), t)
def en_to_de(t): return translate_with_model(get_en_de(), t)
def en_to_ne(t): return translate_with_model(get_en_ne(), t)
def ne_to_en(t): return translate_with_model(get_ne_en(), t)
def de_to_ne(t): return en_to_ne(de_to_en(t))
def ne_to_de(t): return en_to_de(ne_to_en(t))

# -------------------------------------------------------
# Streamlit Interface
# -------------------------------------------------------
mode = st.radio("Select Mode:", [
    "🎤 Audio → Translation (with Speech Output)",
    "🖼️ Image → Text (EasyOCR)",
    "🔤 Text Translation (DE↔EN↔NE)"
])

# ---------------- AUDIO → TRANSLATION ----------------
if mode.startswith("🎤"):
    st.subheader("Upload Audio in German, English, or Nepali")
    target_lang = st.selectbox("Translate To:", ["English", "German", "Nepali"])
    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])

    if audio_file:
        st.audio(audio_file)
        tmp_path = "temp_audio.wav"
        with open(tmp_path, "wb") as f:
            f.write(audio_file.read())

        with st.spinner("🎧 Transcribing audio..."):
            transcribed = transcribe_long_audio_file(tmp_path)
        st.text_area("Transcribed Text:", transcribed, height=200)

        # Auto language detection (simple heuristic)
        lang_detect = "en"
        if any(ch in transcribed for ch in "äöüß"): lang_detect = "de"
        elif any(ord(ch) > 2300 for ch in transcribed): lang_detect = "ne"

        # Choose translation path
        with st.spinner(f"🌐 Translating from {lang_detect.upper()} → {target_lang} ..."):
            if lang_detect == "de" and target_lang == "English":
                translated = de_to_en(transcribed)
            elif lang_detect == "de" and target_lang == "Nepali":
                translated = de_to_ne(transcribed)
            elif lang_detect == "en" and target_lang == "German":
                translated = en_to_de(transcribed)
            elif lang_detect == "en" and target_lang == "Nepali":
                translated = en_to_ne(transcribed)
            elif lang_detect == "ne" and target_lang == "English":
                translated = ne_to_en(transcribed)
            elif lang_detect == "ne" and target_lang == "German":
                translated = ne_to_de(transcribed)
            else:
                translated = transcribed  # same language

        st.success("✅ Translation Complete")
        st.text_area("Translated Text:", translated, height=200)

        # Choose correct TTS language code
        tts_lang = "en"
        if target_lang == "German": tts_lang = "de"
        elif target_lang == "Nepali": tts_lang = "ne"

        # Play translated audio
        tts_html = text_to_speech(translated, tts_lang)
        if tts_html:
            st.markdown(tts_html, unsafe_allow_html=True)
            with open("temp_tts.mp3", "rb") as audio_f:
                st.download_button("⬇️ Download Translated Audio", data=audio_f, file_name="translated_audio.mp3")

# ---------------- OCR ----------------
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # ensures consistent detection

elif mode.startswith("🖼️"):
    st.subheader("Upload Image for OCR & Translation")
    img_file = st.file_uploader("Image", type=["jpg", "jpeg", "png"])

    if img_file:
        img = resize_image(img_file)
        st.image(img, caption="Processed Image (Resized <1MB)", use_column_width=True)

        with st.spinner("🔍 Extracting text from image..."):
            reader = get_ocr()
            text = "\n".join(reader.readtext(np.array(img), detail=0))

        if not text.strip():
            st.warning("No readable text found in the image.")
        else:
            st.success("✅ OCR Complete")
            st.text_area("Extracted Text:", text, height=200)

            # ---------- Accurate Language Detection ----------
            try:
                detected_lang = detect(text)
            except Exception:
                detected_lang = "en"  # fallback if detection fails

            lang_map = {"de": "German", "en": "English", "ne": "Nepali"}
            src_lang = lang_map.get(detected_lang, "English")

            st.markdown(f"**Detected Language:** {src_lang}")

            # ---------- Translation ----------
            target_lang = st.selectbox(
                "Translate To:", ["English", "German", "Nepali"], index=0
            )

            if st.button("Translate Extracted Text"):
                with st.spinner(f"🌐 Translating from {src_lang} → {target_lang}..."):
                    if src_lang == "German" and target_lang == "English":
                        translated = de_to_en(text)
                    elif src_lang == "German" and target_lang == "Nepali":
                        translated = de_to_ne(text)
                    elif src_lang == "English" and target_lang == "German":
                        translated = en_to_de(text)
                    elif src_lang == "English" and target_lang == "Nepali":
                        translated = en_to_ne(text)
                    elif src_lang == "Nepali" and target_lang == "English":
                        translated = ne_to_en(text)
                    elif src_lang == "Nepali" and target_lang == "German":
                        translated = ne_to_de(text)
                    else:
                        translated = text

                st.success("✅ Translation Complete")
                st.text_area("Translated Text:", translated, height=200)

                # ---------- Optional TTS ----------
                tts_lang = "en"
                if target_lang == "German": tts_lang = "de"
                elif target_lang == "Nepali": tts_lang = "ne"

                tts_html = text_to_speech(translated, tts_lang)
                if tts_html:
                    st.markdown(tts_html, unsafe_allow_html=True)
                    with open("temp_tts.mp3", "rb") as audio_f:
                        st.download_button(
                            "⬇️ Download Translated Audio",
                            data=audio_f,
                            file_name="ocr_translation.mp3"
                        )

        clear_memory()

# ---------------- TEXT ----------------
else:
    st.subheader("Text Translation")
    direction = st.selectbox("Translation Direction:", [
        "German → English", "English → German",
        "English → Nepali", "Nepali → English",
        "German → Nepali (via English)", "Nepali → German (via English)"
    ])
    text_in = st.text_area("Enter Text:", height=150)

    if st.button("Translate"):
        if not text_in.strip():
            st.warning("Please enter text first.")
        else:
            with st.spinner("Translating..."):
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
                else:
                    out = ne_to_de(text_in)
            st.success("✅ Translation Complete")
            st.text_area("Translated Text:", out, height=200)

            # TTS output
            tts_lang = "en"
            if "German" in direction: tts_lang = "de"
            elif "Nepali" in direction: tts_lang = "ne"
            tts_html = text_to_speech(out, tts_lang)
            if tts_html:
                st.markdown(tts_html, unsafe_allow_html=True)
                with open("temp_tts.mp3", "rb") as audio_f:
                    st.download_button("⬇️ Download Translated Audio", data=audio_f, file_name="translated_audio.mp3")

st.markdown("---")
st.caption("Optimized for Nepali Students.")
