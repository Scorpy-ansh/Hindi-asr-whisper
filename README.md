<!-- HERO -->
<div align="center">
  <h1>🎤 Hindi-ASR-Whisper</h1>
  <p><b>Hindi speech-to-text with Whisper</b> · fine-tune & evaluate (WER/CER) · <b>Streamlit app</b> with MP3/WAV + mic · <b>Hindi → English</b> translation · batch mode</p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/Scorpy-ansh/Hindi-asr-whisper/stargazers">
      <img alt="Stars" src="https://img.shields.io/github/stars/Scorpy-ansh/Hindi-asr-whisper?style=flat&color=ffd166">
    </a>
    <a href="https://github.com/Scorpy-ansh/Hindi-asr-whisper/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/Scorpy-ansh/Hindi-asr-whisper?style=flat&color=ef476f">
    </a>
    <img alt="Python" src="https://img.shields.io/badge/python-3.10–3.12-118ab2">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ef3d60">
    <img alt="Transformers" src="https://img.shields.io/badge/Transformers-🤗-9b5de5">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-06d6a0">
  </p>

  <!-- ONE-CLICK LIVE APP BUTTON (BIG) -->
  <p>
    <a href="https://hindi-asr-whisper.streamlit.app/" target="_blank">
      <img src="https://img.shields.io/badge/OPEN%20IN%20STREAMLIT-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open in Streamlit">
    </a>
  </p>

  <!-- Optional QR for slides / mobile users -->
  <p>
    <a href="https://hindi-asr-whisper.streamlit.app/" target="_blank">
      <img src="https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=https://hindi-asr-whisper.streamlit.app/" alt="QR to app">
    </a>
  </p>

  <!-- Demo GIF (add later at docs/demo.gif) -->
  <img src="docs/demo.gif" alt="Demo" width="850">
</div>


## Table of Contents
- [✨ Features](#-features)
- [🧠 What’s inside](#-whats-inside)
- [🧩 Architecture](#-architecture)
- [🚀 Quickstart](#-quickstart)
- [🧪 Evaluation](#-evaluation)
- [🖥️ App](#️-app)
- [📊 Results](#-results)
- [📦 Project structure](#-project-structure)
- [🙋‍♀️ FAQ / Viva Short Answers](#️-faq--viva-short-answers)
- [🙏 Credits](#-credits)
- [📝 License](#-license)

---

## ✨ Features
- 🗣️ **Whisper Small (Hindi)** baseline + **fine-tuning** on Common Voice (v22)
- 📈 **WER / CER** with speaker-disjoint test split
- 🧰 Light **Hindi text normalization** (danda/quotes/spacing)
- 🧪 **Feature visualizations** (spectrogram, token/length dists)
- 🌐 **Streamlit** app: MP3/WAV upload **or microphone**, Hindi transcript + **English translation**
- 📦 **Batch mode**: many files → CSV (optionally with references to compute WER/CER)
- ⚙️ Windows-friendly (no TensorFlow/JAX, MP3 via torchaudio)

---

## 🧠 What’s inside
- **Model:** `collabora/whisper-small-hindi` (encoder–decoder Transformer)  
- **Features:** 80-bin **log-Mel** spectrograms @ 16 kHz mono  
- **Training (demo):** AdamW, lr `5e-5`, grad clip `1.0`, small steps for CPU  
- **Translation:** MT (`Helsinki-NLP/opus-mt-hi-en`) or Whisper direct translate

---

## 🧩 Architecture

```mermaid
flowchart LR
    A[🎙️ Audio Input<br>(WAV / MP3 / Mic)] --> B[🔊 torchaudio Load<br>Mono 16 kHz Resample]
    B --> C[📈 Log-Mel Spectrogram<br>(80 × T Features)]
    C --> D[🧠 Whisper Encoder]
    D --> E[🗣️ Whisper Decoder]
    E --> F[📝 Hindi Transcript]
    F --> G{🌐 Translation Mode}
    G -->|MT hi→en| H1[🇬🇧 English Text (MT)]
    G -->|Whisper translate| H2[🇬🇧 English Text (Whisper)]
    F --> I[📊 WER / CER vs Reference]

## ⚙️ Data Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as Streamlit App
    participant Whisper as Whisper ASR
    participant Translator as MT Model
    participant Output as Result UI

    User->>Streamlit: 🎙 Upload or record audio
    Streamlit->>Whisper: 🔊 Send waveform for transcription
    Whisper-->>Streamlit: 📝 Hindi transcript
    Streamlit->>Translator: 🌐 Translate Hindi→English
    Translator-->>Streamlit: 🇬🇧 English text
    Streamlit-->>User: 📊 Display transcript, translation, WER

---

This pairing gives a **complete, professional view**:
- *Architecture* → shows static system design  
- *Sequence Diagram* → shows runtime interaction  

Both render visually on GitHub with good readability and clarity for users exploring your repo.
