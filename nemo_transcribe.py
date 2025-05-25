import soundfile as sf
import numpy as np
from nemo.collections.asr.models import EncDecCTCModelBPE

# Path to your .nemo model
MODEL_PATH = "/Users/saani/Mvaak AI/stt_hi_conformer_ctc_medium.nemo"
# Path to your audio file
AUDIO_PATH = "/Users/saani/Mvaak AI/audio3.wav"

# Load the model
asr_model = EncDecCTCModelBPE.restore_from(MODEL_PATH)
asr_model.eval()

# Read the audio file
audio, sr = sf.read(AUDIO_PATH)
if sr != 16000:
    raise ValueError(f"Sample rate must be 16kHz, got {sr}")

# If stereo, convert to mono
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Transcribe
transcription = asr_model.transcribe([audio])
print("Transcription:", transcription[0])

print(asr_model.cfg.preprocessor)


