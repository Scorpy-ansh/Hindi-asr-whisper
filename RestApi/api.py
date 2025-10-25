import io, os, re, time
from typing import Optional
import numpy as np
import soundfile as sf
import torch, torchaudio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import pipeline
from jiwer import wer

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

app = FastAPI(title="Hindi ASR API")

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

def load_audio_from_bytes(audio_bytes: bytes, target_sr=16000):
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if wav.shape[1] > 1:
        wav = np.mean(wav, axis=1, dtype=np.float32)
    else:
        wav = wav[:, 0]
    wav, sr = _resample_if_needed(wav, sr, target_sr)
    return {"array": wav, "sampling_rate": sr}

DEVICE = 0 if torch.cuda.is_available() else -1
_ASR = None
_TRANS = None

def get_asr(model_choice: str = "baseline"):
    global _ASR
    if _ASR is not None:
        return _ASR
  
    model_id = "collabora/whisper-small-hindi" if model_choice == "baseline" else "models/whisper_small_hi_ft"
    _ASR = pipeline("automatic-speech-recognition", model=model_id, device=DEVICE)
    return _ASR

def get_translator():
    global _TRANS
    if _TRANS is not None:
        return _TRANS
    try:
        _TRANS = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en", device=DEVICE)
    except Exception:
        _TRANS = None
    return _TRANS

def translate_long(text, translator, max_chars=400):
    parts, s = [], text
    while len(s) > max_chars:
        cut = s.rfind(" ", 0, max_chars)
        cut = cut if cut != -1 else max_chars
        parts.append(s[:cut]); s = s[cut:].lstrip()
    parts.append(s)
    outs = translator(parts)
    return " ".join(o["translation_text"] for o in outs)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": "gpu" if DEVICE == 0 else "cpu"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    do_norm: bool = Form(True),
    translate_mode: str = Form("mt"),      
    ref_text: Optional[str] = Form(None)    
):
    audio_bytes = await file.read()
    audio = load_audio_from_bytes(audio_bytes, target_sr=16000)

    asr = get_asr("baseline")

    t0 = time.time()
    out = asr(audio, generate_kwargs={"language": "hi", "task": "transcribe"})
    dt = time.time() - t0

    hyp_hi = out["text"]
    hyp_norm = normalize_hi(hyp_hi) if do_norm else hyp_hi

    en_text = None
    if translate_mode == "mt":
        trans = get_translator()
        if trans is None:
            out_en = asr(audio, generate_kwargs={"language": "hi", "task": "translate"})
            en_text = out_en["text"]
        else:
            text_for_mt = hyp_norm if do_norm else hyp_hi
            en_text = translate_long(text_for_mt, trans)
    else:
        out_en = asr(audio, generate_kwargs={"language": "hi", "task": "translate"})
        en_text = out_en["text"]

    wer_score = None
    if ref_text and ref_text.strip():
        ref = normalize_hi(ref_text) if do_norm else ref_text
        wer_score = float(wer([ref], [hyp_norm]))

    return JSONResponse({
        "transcript_hi": hyp_hi,
        "transcript_hi_normalized": hyp_norm if do_norm else None,
        "translation_en": en_text,
        "inference_time_s": round(dt, 3),
        "device": "gpu" if DEVICE == 0 else "cpu",
        "wer": wer_score
    })
