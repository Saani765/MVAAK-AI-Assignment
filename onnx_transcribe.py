import soundfile as sf
import numpy as np
import librosa
import onnxruntime
import torch
from nemo.collections.asr.models import EncDecCTCModelBPE

# Paths
ONNX_PATH = "/Users/saani/Mvaak AI/stt_hi_conformer_ctc_medium_simplified.onnx"
AUDIO_PATH = "/Users/saani/Mvaak AI/audio2_16khz.wav"

# Your vocab (should be 129, last is blank)
LABELS = [ '<unk>', 'ा', 'र', 'ी', '▁', 'े', 'न', 'ि', 'त', 'क', '्', 'ल', 'म', 'स', 'ं', '▁स', 'ह', 'ो', 'ु', 'द', 'य', 'प', '▁है', '▁के', 'ग', '▁ब', '▁म', 'व', '▁क', '▁में', 'ट', '▁अ', 'ज', '▁द', '▁प', '▁आ', '्र', 'ू', '▁ज', '▁की', '▁र', 'ध', 'र्', 'ों', 'ख', '▁का', '्य', 'च', 'ए', 'ब', 'भ', 'ने', '▁को', '▁से', '▁ल', '▁और', '▁प्र', '▁त', '▁कर', '▁व', 'ता', 'श', '▁कि', '▁ह', '▁न', '▁ग', 'ना', '▁हो', 'ै', '▁पर', 'थ', '▁उ', 'ड', '▁च', 'िक', 'ण', 'ई', '▁हैं', 'िया', '▁इस', 'फ', '▁वि', 'वा', '▁जा', 'ष', 'ित', '▁श', 'ें', '▁ने', 'ेश', 'ते', 'इ', '▁भी', 'का', '▁एक', '्या', '▁हम', '▁सं', 'िल', 'ंग', 'ड़', 'छ', 'क्ष', 'ौ', 'ठ', '़', 'ॉ', 'ओ', 'ढ', 'घ', 'आ', 'झ', 'ऐ', 'ँ', 'ऊ', 'उ', 'ः', 'औ', ',', 'ऍ', 'ॅ', 'ॠ', 'ऋ', 'ऑ', 'ञ', 'ृ', 'अ', 'ङ', '<blank>' ]

# Preprocessing (match NeMo config)
def compute_mel_spectrogram(audio_data, samplerate):
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    n_fft = 512
    win_length = int(0.025 * samplerate)
    hop_length = int(0.01 * samplerate)
    n_mels = 80
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=samplerate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        n_mels=n_mels,
        fmin=0,
        fmax=None,
        power=2.0,
        center=True,
        pad_mode='reflect'
    )
    log_mel_spec = np.log(mel_spec + 1e-9)
    # Per-feature normalization
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec, axis=0)) / (np.std(log_mel_spec, axis=0) + 1e-9)
    audio_data = audio_data + np.random.normal(0, 1e-5, audio_data.shape)
    print("Mel-spectrogram shape:", log_mel_spec.shape)
    return log_mel_spec.T

# CTC Greedy Decode
def ctc_greedy_decode(logprobs, labels):
    result = np.argmax(logprobs, axis=2)[:, 0]
    decoded_text = ""
    last_token_idx = -1
    blank_idx = len(labels) - 1
    for token_idx in result:
        if token_idx >= len(labels):
            print(f"Warning: token_idx {token_idx} out of range for labels of length {len(labels)}")
            continue
        if token_idx != blank_idx and token_idx != last_token_idx:
            decoded_text += labels[token_idx]
        last_token_idx = token_idx
    print("Decoded indices:", result)
    return decoded_text

# Load audio
audio_data, samplerate = sf.read(AUDIO_PATH)
audio_data = audio_data.astype(np.float32)

# Preprocess
mel_spec = compute_mel_spectrogram(audio_data, samplerate)
processed_audio = np.expand_dims(mel_spec.T, axis=0)
processed_length = np.array([mel_spec.shape[0]], dtype=np.int64)

# ONNX inference
session = onnxruntime.InferenceSession(ONNX_PATH)
input_names = [i.name for i in session.get_inputs()]
output_names = [o.name for o in session.get_outputs()]
input_feed = {
    input_names[0]: processed_audio,
    input_names[1]: processed_length
}
outputs = session.run(output_names, input_feed)
logprobs = outputs[0]

# Decode
transcribed_text = ctc_greedy_decode(logprobs, LABELS)
print("Transcription:", transcribed_text)

# --- NeMo feature extraction ---
NEMO_MODEL_PATH = "/Users/saani/Mvaak AI/stt_hi_conformer_ctc_medium.nemo"
asr_model = EncDecCTCModelBPE.restore_from(NEMO_MODEL_PATH)
asr_model.eval()

audio_tensor = torch.tensor(audio_data).unsqueeze(0)
length_tensor = torch.tensor([audio_data.shape[0]])
with torch.no_grad():
    nemo_features, nemo_length = asr_model.preprocessor(input_signal=audio_tensor, length=length_tensor)
print("NeMo features shape:", nemo_features.shape)
print("NeMo features mean/std:", nemo_features.mean().item(), nemo_features.std().item())

# --- ONNX pipeline features ---
mel_spec = compute_mel_spectrogram(audio_data, samplerate)
print("ONNX pipeline features shape:", mel_spec.shape)
print("ONNX pipeline features mean/std:", mel_spec.mean(), mel_spec.std())

print(session.get_inputs()[0].shape)
