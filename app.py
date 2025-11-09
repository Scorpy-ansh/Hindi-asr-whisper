import os
import io
import re
import time
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
import torch
import torchaudio
import numpy as np
import soundfile as sf
from transformers import pipeline

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

st.set_page_config(page_title="Hindi ASR", layout="centered")

def normalize_hi(s: str) -> str:
    s = "" if s is None else unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("।", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^\s\u0900-\u097F0-9]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sanitize_for_mt(hin: str) -> str:
    s = "" if hin is None else unicodedata.normalize("NFKC", str(hin))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def postprocess_english(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    if not re.search(r"[.?!]$", t):
        t = t + "."
    if len(t) > 0:
        t = t[0].upper() + t[1:]
    return t


def _to_mono_and_resample(wav_arr: np.ndarray, sr: int, target_sr: int = 16000):
    if wav_arr.ndim == 1:
        mono = wav_arr.astype(np.float32)
    else:
        mono = np.mean(wav_arr, axis=1).astype(np.float32)
    if sr != target_sr:
        wav_t = torch.from_numpy(mono).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        with torch.no_grad():
            wav_t = resampler(wav_t)
        mono = wav_t.squeeze(0).numpy().astype(np.float32)
        sr = target_sr
    return mono, sr

def load_audio(file_obj, target_sr: int = 16000):
    data = file_obj.read()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono, sr = _to_mono_and_resample(wav, sr, target_sr)
    return {"array": mono, "sampling_rate": sr}

def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000):
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono, sr = _to_mono_and_resample(wav, sr, target_sr)
    return {"array": mono, "sampling_rate": sr}


DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_asr(model_choice: str):
    local_candidates = [
        "models/whisper_small_hi_ft",
        "models/whisper small hi ft",
        "models/whisper_small_hi",
        "models/whisper-hi-ft",
    ]
    if model_choice == "Fine-tuned (local)":
        model_id = None
        for p in local_candidates:
            if os.path.isdir(p) and len(os.listdir(p)) > 2:
                model_id = p
                break
        if model_id is None:
            st.warning("Fine-tuned model folder not found. Falling back to baseline.")
            model_id = "collabora/whisper-small-hindi"
        else:
            st.success(f"Fine-tuned model found at: {model_id}")
    else:
        model_id = "collabora/whisper-small-hindi"

    try:
        return pipeline("automatic-speech-recognition", model=model_id, device=DEVICE)
    except Exception as e:
        st.error(f"Failed to load ASR pipeline: {e}")
        if model_id != "collabora/whisper-small-hindi":
            return pipeline("automatic-speech-recognition", model="collabora/whisper-small-hindi", device=DEVICE)
        raise

@st.cache_resource
def load_helsinki_translator():
    try:
        return pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en", device=DEVICE)
    except Exception:
        return None

@st.cache_resource
def load_strong_translator():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception:
        return None
    model_name = "facebook/m2m100_418M"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(TORCH_DEVICE)
        return (tokenizer, model)
    except Exception:
        return None

def translate_hi_to_en_strong(texts, tokenizer_model_tuple):
    if tokenizer_model_tuple is None:
        return ""
    tokenizer, model = tokenizer_model_tuple
    device = model.device
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    out = model.generate(**inputs, num_beams=6, length_penalty=1.0, early_stopping=True, max_new_tokens=512)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    return decoded[0] if decoded else ""

def translate_long_parts(text: str, translator_pipeline, max_chars: int = 400):
    if translator_pipeline is None:
        return ""
    parts, s = [], text
    while len(s) > max_chars:
        cut = s.rfind(" ", 0, max_chars)
        cut = cut if cut != -1 else max_chars
        parts.append(s[:cut])
        s = s[cut:].lstrip()
    parts.append(s)
    outs = translator_pipeline(parts)
    return " ".join(o.get("translation_text", "") for o in outs)

st.title("Hindi Speech-to-Text System")

with st.sidebar:
    model_choice = st.selectbox("Model", ["Fine-tuned (local)", "Baseline (pretrained)"])
    do_norm = st.checkbox("Normalize Hindi text for WER", value=True)
    trans_mode = st.selectbox(
        "Translation strategy",
        ["Whisper direct (audio→en)", "Helsinki MT (hi→en)", "Both (Whisper+Helsinki)", "Strong (m2m100 beam)"]
    )
    st.caption("Tip: 16 kHz mono recommended. Heavy translators require GPU & memory.")

uploaded = st.file_uploader("Upload audio (WAV or MP3)", type=["wav", "mp3"])
st.caption("Or record below and click again to stop")
try:
    from audio_recorder_streamlit import audio_recorder
    audio_bytes = audio_recorder(text="Click to record / stop")
except Exception:
    audio_bytes = None

if audio_bytes is not None and len(audio_bytes) == 0:
    audio_bytes = None

ref_text = st.text_input("Optional: paste ground-truth transcript to compute WER")

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
        audio = load_audio_from_bytes(audio_bytes)

    with st.spinner("Transcribing audio..."):
        t0 = time.time()
        try:
            asr_out = asr(audio, generate_kwargs={"language": "hi", "task": "transcribe"})
            if isinstance(asr_out, dict):
                hyp_hi = asr_out.get("text", "") or ""
            elif isinstance(asr_out, (list, tuple)) and len(asr_out) > 0:
                hyp_hi = asr_out[0].get("text", "") or ""
            else:
                hyp_hi = ""
        except Exception:
            hyp_hi = ""
        dt = time.time() - t0

    hyp_display = normalize_hi(hyp_hi) if do_norm else hyp_hi

    st.markdown("### Transcript (Hindi)")
    st.write(hyp_display or "_No transcription produced_")
    st.caption(f"Inference time: {dt:.2f}s on {'GPU' if DEVICE == 0 else 'CPU'}")

    # ---------- Translation ----------
    helsinki = None
    strong_tokmdl = None
    if trans_mode in ("Helsinki MT (hi→en)", "Both (Whisper+Helsinki)"):
        helsinki = load_helsinki_translator()
    if trans_mode == "Strong (m2m100 beam)":
        strong_tokmdl = load_strong_translator()

    text_for_mt = sanitize_for_mt(hyp_hi) if hyp_hi else ""

    whisper_en = ""
    if trans_mode in ("Whisper direct (audio→en)", "Both (Whisper+Helsinki)"):
        try:
            raw_tensor = torch.tensor(audio["array"])
            out_whisper_en = asr({"raw": raw_tensor, "sampling_rate": audio["sampling_rate"]}, generate_kwargs={"language": "hi", "task": "translate"})
            if isinstance(out_whisper_en, dict):
                whisper_en = out_whisper_en.get("text", "") or ""
            elif isinstance(out_whisper_en, (list, tuple)) and len(out_whisper_en) > 0:
                whisper_en = out_whisper_en[0].get("text", "") or ""
            else:
                whisper_en = ""
        except Exception:
            whisper_en = ""

    if not whisper_en:
        if helsinki is None:
            helsinki = load_helsinki_translator()
        if helsinki is not None:
            try:
                whisper_en = translate_long_parts(text_for_mt, helsinki)
            except Exception:
                whisper_en = ""

    # Strong translator option
    m2m_en = ""
    if strong_tokmdl is not None:
        try:
            m2m_en = translate_hi_to_en_strong([text_for_mt], strong_tokmdl) if text_for_mt.strip() else ""
        except Exception:
            m2m_en = ""

    # Postprocess outputs
    whisper_en = postprocess_english(whisper_en)
    helsinki_en = postprocess_english(translate_long_parts(text_for_mt, helsinki) if (trans_mode in ("Helsinki MT (hi→en)","Both (Whisper+Helsinki)")) else "")
    m2m_en = postprocess_english(m2m_en)

    st.markdown("---")
    col_hi, col_en = st.columns([1, 1])

    with col_hi:
        st.markdown("#### Hindi")
        st.write(hyp_display or "_No transcription_")
        if ref_text and ref_text.strip():
            try:
                from jiwer import wer
                ref_eval = normalize_hi(ref_text) if do_norm else ref_text.strip()
                hyp_eval = normalize_hi(hyp_hi) if do_norm else hyp_hi
                score = wer([ref_eval], [hyp_eval])
                st.metric("WER", f"{score:.3f}")
            except Exception:
                pass

    with col_en:
        st.markdown("#### English")
        # choose display based on mode
        if trans_mode == "Whisper direct (audio→en)":
            display_text = whisper_en
        elif trans_mode == "Helsinki MT (hi→en)":
            display_text = helsinki_en
        elif trans_mode == "Both (Whisper+Helsinki)":
            # prefer whisper if present, else helsinki
            display_text = whisper_en or helsinki_en
        elif trans_mode == "Strong (m2m100 beam)":
            display_text = m2m_en
        else:
            display_text = whisper_en or helsinki_en or m2m_en

        st.write(display_text or "_No translation available_")

        combined = f"Hindi: {hyp_display}\n\nEnglish: {display_text or ''}"
        st.download_button("Download Translation (.txt)", combined.encode("utf-8"), "transcript_en.txt")

        try:
            from gtts import gTTS
            if (display_text or "").strip():
                tmp = NamedTemporaryFile(delete=False, suffix=".mp3")
                tts = gTTS(text=display_text, lang="en")
                tts.save(tmp.name)
                st.audio(tmp.name)
        except Exception:
            pass

else:
    st.info("Upload a WAV/MP3 or record audio. Choose model and translation mode in the sidebar.")
