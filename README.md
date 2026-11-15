# German Tutor — Multimodal Language Translation Prototype

German Tutor is an academic AI application developed as a final-examination project. It explores how speech recognition, machine translation, optical character recognition (OCR), and text-to-speech can be combined in one Streamlit interface to support language access for Nepali learners.

> **Project status:** Academic prototype. The repository documents the original project and its current capabilities; it is not presented as a production-ready language-learning platform.

## What it does

- **Text translation:** German ↔ English, English ↔ Nepali, and German ↔ Nepali (through English).
- **Audio transcription:** Accepts MP3, WAV, and M4A uploads and uses OpenAI Whisper (base) for speech-to-text.
- **Image text extraction:** Uses EasyOCR to extract text from JPG, JPEG, and PNG images.
- **Language detection:** Uses langdetect in the OCR workflow and a basic heuristic in the audio workflow.
- **Speech output:** Uses Google Text-to-Speech (gTTS) to generate downloadable translated audio.

## How it works

\`\`\`mermaid
flowchart TD
    A[User input] --> B{Input type}
    B -->|Text| C[Translation models]
    B -->|Audio| D[Whisper speech recognition]
    D --> C
    B -->|Image| E[EasyOCR text extraction]
    E --> F[Language detection]
    F --> C
    C --> G[Translated text]
    G --> H[gTTS speech output]
\`\`\`

## Models and tools

| Component | Technology |
|---|---|
| App interface | Streamlit |
| Speech recognition | OpenAI Whisper base |
| German ↔ English translation | Helsinki-NLP OPUS-MT |
| English ↔ Nepali translation | Hugging Face MarianMT models |
| Image OCR | EasyOCR |
| Language detection | langdetect |
| Text-to-speech | gTTS |
| Core language | Python |

## Run locally

### 1. Clone the repository

\`\`\`bash
git clone https://github.com/raverer/German-Tutor.git
cd German-Tutor
\`\`\`

### 2. Create and activate a virtual environment

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

On Windows, activate it with:

\`\`\`powershell
.venv\\Scripts\\activate
\`\`\`

### 3. Install dependencies

\`\`\`bash
python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

Whisper also requires **FFmpeg** to be installed and available on your system path. On macOS with Homebrew:

\`\`\`bash
brew install ffmpeg
\`\`\`

### 4. Start the app

\`\`\`bash
streamlit run app.py
\`\`\`

The first run may take time because the required Hugging Face models are downloaded.

## Current limitations and next improvements

This repository is an academic prototype, and several areas need further engineering before production use:

- The audio transcription path currently passes an English-language hint to Whisper; German and Nepali audio need proper language selection or multilingual handling.
- OCR is initialised with an English reader, so recognition quality for other scripts/languages is not guaranteed.
- Audio language detection uses a simple character-based heuristic; it should be replaced with a reliable language-identification workflow.
- German ↔ Nepali translation is performed through English, which can introduce meaning loss.
- The current Streamlit application contains multiple mode selectors and would benefit from a unified navigation flow and modular code structure.
- Temporary audio files and error handling need improvement for concurrent and deployed use.
- Translation quality, OCR accuracy, and transcription quality have not yet been benchmarked systematically.

## Academic context

Built as a final-examination project to explore practical integration of pretrained NLP and computer-vision models into an accessible application. The project helped develop hands-on experience with Python, Streamlit, Hugging Face Transformers, speech processing, OCR, and model integration.

## Disclaimer

Machine-generated translations and transcriptions may contain errors. Do not rely on this prototype for legal, medical, immigration, or other high-stakes translation.

## Repository

[https://github.com/raverer/German-Tutor](https://github.com/raverer/German-Tutor)
