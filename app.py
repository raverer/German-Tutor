import streamlit as st
import torch, gc, io, librosa, numpy as np
from PIL import Image
import easyocr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------
st.set_page_config(page_title="Multimodal Translator", layout="centered")
st.title("🎧🖼️ Multimodal Translator")
st.caption("Whisper-Base (chunked) + EasyOCR + English→Nepali Translation — Optimized for Streamlit Cloud")

# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------
def clear_memory():
    """Free up memory safely after each mode."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def resize_image(uploaded_file):
    """Resize uploaded image to under ~1MB and within 1280px."""
    img = Image.open(uploaded_file)
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return Image.open(buf)

# -------------------------------------------------------
# Cached Model Loaders (Lazy Loading)
# -------------------------------------------------------
@st.cache_resource
def load_whisper():
    """Load Whisper Base ASR model."""
    return pipeline("automatic-speech-recognition", model="openai/whisper-base")

@st.cache_resource
def load_ocr():
    """Load EasyOCR (English)."""
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_translation():
    """Load English→Nepali Translation Model."""
    model = AutoModelForSeq2SeqLM.from_pretrained("Hemg/english-To-Nepali-TRanslate")
    tokenizer = AutoTokenizer.from_pretrained("Hemg/english-To-Nepali-TRanslate")
    return model, tokenizer

# -------------------------------------------------------
# Whisper Chunking Transcription
# -------------------------------------------------------
def transcribe_long_audio(file_path, chunk_length_s=30, overlap_s=2):
    """Split long audio into 30s chunks and transcribe sequentially."""
    asr = load_whisper()
    audio, sr = librosa.load(file_path, sr=16000)
    chunk_samples = int(chunk_length_s * sr)
    overlap_samples = int(overlap_s * sr)
    transcriptions = []

    for start in range(0, len(audio), chunk_samples - overlap_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        chunk_input = {"array": chunk, "sampling_rate": sr}
        result = asr(chunk_input)
        transcriptions.append(result["text"])
        if end == len(audio):
            break

    del asr
    clear_memory()
    return " ".join(transcriptions)

# -------------------------------------------------------
# UI Mode Selector
# -------------------------------------------------------
mode = st.radio("Select Mode:", [
    "🎤 Audio → Text (Whisper-Base)",
    "🖼️ Image → Text (EasyOCR)",
    "🈶 English → Nepali Translation"
])

# -------------------------------------------------------
# AUDIO MODE — Whisper Base (Chunked)
# -------------------------------------------------------
if mode == "🎤 Audio → Text (Whisper-Base)":
    st.subheader("Upload Audio File (MP3/WAV/M4A)")
    audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])

    if audio_file is not None:
        st.audio(audio_file)
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())

        with st.spinner("🔹 Transcribing audio with Whisper Base (chunked)..."):
            text = transcribe_long_audio("temp_audio.wav")
            st.success("✅ Transcription Complete!")
            st.text_area("Transcribed Text:", text, height=200)

# -------------------------------------------------------
# IMAGE MODE — EasyOCR
# -------------------------------------------------------
elif mode == "🖼️ Image → Text (EasyOCR)":
    st.subheader("Upload Image")
    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        img = resize_image(img_file)
        st.image(img, caption="Processed Image (Resized <1MB)", use_container_width=True)

        with st.spinner("🔹 Extracting text using EasyOCR..."):
            reader = load_ocr()
            result = reader.readtext(np.array(img), detail=0)
            text = "\n".join(result)
            st.success("✅ OCR Complete!")
            st.text_area("Extracted Text:", text, height=200)

        del reader
        clear_memory()

# -------------------------------------------------------
# TRANSLATION MODE — English → Nepali
# -------------------------------------------------------
else:
    st.subheader("English → Nepali Translation")
    text_input = st.text_area("Enter English text:")

    if st.button("Translate"):
        if text_input.strip():
            with st.spinner("🔹 Translating..."):
                model, tokenizer = load_translation()
                inputs = tokenizer(text_input, return_tensors="pt")
                outputs = model.generate(**inputs, max_new_tokens=100)
                translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.success("✅ Translation Complete!")
                st.text_area("Nepali Translation:", translated_text, height=150)

            del model, tokenizer
            clear_memory()
        else:
            st.warning("Please enter some English text first!")

st.markdown("---")
st.caption("Built with ❤️ by Rajib Rawal using Streamlit.")
