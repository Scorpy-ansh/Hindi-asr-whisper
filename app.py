import os, tempfile, time, re, io
from pathlib import Path
import streamlit as st
import torch, torchaudio
import numpy as np
import soundfile as sf
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

st.set_page_config(page_title="Hindi ASR", page_icon="🎤", layout="centered")

def normalize_hi(s: str) -> str:
    s = str(s)
    s = s.replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"')
    s = s.replace("।"," ").lower()
    s = re.sub(r"[^ \u0900-\u097F0-9a-z]", " ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

def _resample_if_needed(wav_1d: np.ndarray, sr: int, target_sr=16000):
    if sr == target_sr:
        return wav_1d.astype(np.float32), sr
    wav_t = torch.from_numpy(wav_1d).float().unsqueeze(0)
    wav_t = torchaudio.transforms.Resample(sr, target_sr)(wav_t)
    return wav_t.squeeze(0).numpy().astype(np.float32), target_sr

def load_audio(file, target_sr=16000):
    # Decode with soundfile (works for wav/mp3/flac when libsndfile is available)
    data = file.read()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    # Downmix to mono
    if wav.shape[1] > 1:
        wav = np.mean(wav, axis=1, dtype=np.float32)
    else:
        wav = wav[:, 0]
    wav, sr = _resample_if_needed(wav, sr, target_sr)
    return {"array": wav, "sampling_rate": sr}

def load_audio_from_bytes(audio_bytes: bytes, target_sr=16000, suffix=".wav"):
    # Decode mic bytes with soundfile
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if wav.shape[1] > 1:
        wav = np.mean(wav, axis=1, dtype=np.float32)
    else:
        wav = wav[:, 0]
    wav, sr = _resample_if_needed(wav, sr, target_sr)
    return {"array": wav, "sampling_rate": sr}

DEVICE = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_asr(model_choice: str):
    if model_choice == "Fine-tuned (local)" and os.path.isdir("models/whisper_small_hi_ft"):
        model_id = "models/whisper_small_hi_ft"
    else:
        if model_choice == "Fine-tuned (local)":
            st.warning("Fine-tuned model folder not found. Falling back to baseline.")
        model_id = "collabora/whisper-small-hindi"
    return pipeline("automatic-speech-recognition", model=model_id, device=DEVICE)

# --- FIXED TRANSLATOR FUNCTION ---
@st.cache_resource
def load_translator():
    try:
        return pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en", device=DEVICE)
    except Exception as e:
        st.warning(f"Translation model failed to load ({e}). Using Whisper for translation instead.")
        return None

def translate_long(text, translator, max_chars=400):
    parts, s = [], text
    while len(s) > max_chars:
        cut = s.rfind(" ", 0, max_chars)
        cut = cut if cut != -1 else max_chars
        parts.append(s[:cut]); s = s[cut:].lstrip()
    parts.append(s)
    outs = translator(parts)
    return " ".join(o["translation_text"] for o in outs)

st.title("🎤 Hindi Speech-to-Text (WAV / MP3)")

with st.sidebar:
    model_choice = st.selectbox("Model", ["Fine-tuned (local)", "Baseline (pretrained)"])
    do_norm = st.checkbox("Normalize Hindi text", value=True)
    trans_mode = st.selectbox("English translation mode", ["MT (hi→en from transcript)", "Whisper direct translate (audio→en)"])
    st.caption("Tip: 16 kHz mono gives best results. MP3 works too.")

uploaded = st.file_uploader("Upload audio (WAV or MP3)", type=["wav", "mp3"])
st.caption("Or record below and click again to stop")
audio_bytes = audio_recorder(text="🎙️ Click to record / stop", recording_color="#e74c3c", neutral_color="#2ecc71", icon_size="2x")

# Guard empty mic bytes
if audio_bytes is not None and len(audio_bytes) == 0:
    audio_bytes = None

ref_text = st.text_input("Optional: paste the ground-truth transcript to compute WER")

uploaded_ok = uploaded is not None
bytes_ok = isinstance(audio_bytes, (bytes, bytearray)) and len(audio_bytes) > 0
have_input = uploaded_ok or bytes_ok

if have_input:
    asr = load_asr(model_choice)
    if uploaded_ok:
        st.audio(uploaded)
        audio = load_audio(uploaded)
    else:
        st.audio(audio_bytes, format="audio/wav")
        audio = load_audio_from_bytes(audio_bytes, suffix=".wav")

    with st.spinner("Transcribing..."):
        t0 = time.time()
        out = asr(audio, generate_kwargs={"language": "hi", "task": "transcribe"})
        dt = time.time() - t0

    hyp_hi = out["text"]
    hyp_show = normalize_hi(hyp_hi) if do_norm else hyp_hi
    st.markdown("**Transcript (Hindi):**")
    st.write(hyp_show)
    st.caption(f"Inference time: {dt:.2f}s on {'GPU' if DEVICE==0 else 'CPU'}")

    st.markdown("---")
    st.subheader("English translation")

    # --- FIXED TRANSLATION BLOCK ---
    if trans_mode.startswith("MT"):
        trans = load_translator()
        if trans is None:
            with st.spinner("Translating with Whisper (audio→en)…"):
                out_en = asr(audio, generate_kwargs={"language": "hi", "task": "translate"})
                st.write(out_en["text"])
        else:
            with st.spinner("Translating (MT)…"):
                text_for_mt = normalize_hi(hyp_hi) if do_norm else hyp_hi
                en_text = translate_long(text_for_mt, trans)
            st.write(en_text)
    else:
        with st.spinner("Translating with Whisper (audio→en)…"):
            out_en = asr(audio, generate_kwargs={"language": "hi", "task": "translate"})
            st.write(out_en["text"])

    if ref_text and ref_text.strip():
        try:
            from jiwer import wer
            ref = normalize_hi(ref_text) if do_norm else ref_text
            score = wer([ref], [hyp_show])
            st.metric("WER", f"{score:.3f}")
        except Exception as e:
            st.warning(f"Could not compute WER: {e}")
else:
    st.info("Upload a WAV or MP3, or use the microphone to record.")
