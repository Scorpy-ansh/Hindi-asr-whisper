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

  <!-- Demo GIF (add later at docs/demo.gif) -->
  <img src="docs/demo.gif" alt="Demo" width="850">
</div>

---

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
  A[Audio (WAV/MP3 or Mic)] --> B[torchaudio load<br/>mono + 16 kHz]
  B --> C[Log-Mel (80×T)]
  C --> D[Whisper Encoder]
  D --> E[Whisper Decoder]
  E --> F[Hindi Transcript]
  F -->|optional| G[MT hi→en or Whisper translate]
  G --> H[English Text]
  F --> I[WER/CER vs Reference]
