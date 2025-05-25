from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import onnxruntime
import soundfile as sf
import io
import numpy as np
from scipy.signal import stft
import librosa
import onnx
from onnxruntime.tools import optimizer
import torch
from nemo.collections.asr.models import EncDecCTCModelBPE
# For a proper mel-spectrogram, you would typically use a library like librosa or torchaudio.
# This is a simplified example using scipy and numpy which might not perfectly match NeMo's preprocessing.


LABELS=['<unk>', 'ा', 'र', 'ी', '▁', 'े', 'न', 'ि', 'त', 'क', '्', 'ल', 'म', 'स', 'ं', '▁स', 'ह', 'ो', 'ु', 'द', 'य', 'प', '▁है', '▁के', 'ग', '▁ब', '▁म', 'व', '▁क', '▁में', 'ट', '▁अ', 'ज', '▁द', '▁प', '▁आ', '्र', 'ू', '▁ज', '▁की', '▁र', 'ध', 'र्', 'ों', 'ख', '▁का', '्य', 'च', 'ए', 'ब', 'भ', 'ने', '▁को', '▁से', '▁ल', '▁और', '▁प्र', '▁त', '▁कर', '▁व', 'ता', 'श', '▁कि', '▁ह', '▁न', '▁ग', 'ना', '▁हो', 'ै', '▁पर', 'थ', '▁उ', 'ड', '▁च', 'िक', 'ण', 'ई', '▁हैं', 'िया', '▁इस', 'फ', '▁वि', 'वा', '▁जा', 'ष', 'ित', '▁श', 'ें', '▁ने', 'ेश', 'ते', 'इ', '▁भी', 'का', '▁एक', '्या', '▁हम', '▁सं', 'िल', 'ंग', 'ड़', 'छ', 'क्ष', 'ौ', 'ठ', '़', 'ॉ', 'ओ', 'ढ', 'घ', 'आ', 'झ', 'ऐ', 'ँ', 'ऊ', 'उ', 'ः', 'औ', ',', 'ऍ', 'ॅ', 'ॠ', 'ऋ', 'ऑ', 'ञ', 'ृ', 'अ', 'ङ','<blank>']

app = FastAPI()

# Placeholder for ONNX model loading
# You need to replace 'path/to/your/model.onnx' with the actual path to your ONNX model file.
# Ensure this model file is available in the container.
ONNX_MODEL_PATH = "./asr_hi_conformer.onnx" # Assuming the model is in the same directory and named correctly
onnx_session = None

async def load_model():
    global onnx_session
    if onnx_session is None:
        try:
            onnx_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
            print("ONNX model loaded successfully.")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            # Depending on requirements, you might want to raise an exception or handle this differently
            # For now, we'll allow the app to start but transcription will fail.

@app.on_event("startup")
async def startup_event():
    await load_model()

# Simplified Mel-Spectrogram Calculation (Approximation)
def compute_mel_spectrogram(audio_data, samplerate):
    # Ensure mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    n_fft = 512
    win_length = int(0.025 * samplerate)  # 400
    hop_length = int(0.01 * samplerate)   # 160
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
    # Per-feature normalization (mean-variance normalization)
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec, axis=0)) / (np.std(log_mel_spec, axis=0) + 1e-9)
    audio_data = audio_data + np.random.normal(0, 1e-5, audio_data.shape)
    return log_mel_spec.T  # (time, n_mels)

# Basic CTC Decoding (Greedy)
def ctc_greedy_decode(logprobs, labels):
    # logprobs shape: (time, batch_size, num_labels)
    # Assuming batch_size is 1
    # labels is a list of characters/tokens corresponding to the model's output indices
    # (e.g., ['a', 'b', ..., ' ', '_'] where '_' is the blank token)

    # Get the most likely token index for each time step
    # result shape: (time, batch_size)
    result = np.argmax(logprobs, axis=2)

    # Flatten the result for batch_size 1
    result = result[:, 0]
    print("Result:", result)

    # Decode using greedy approach
    decoded_text = ""
    last_token_idx = -1 # Use -1 to represent no last token
    blank_idx = len(labels) - 1 # Assuming blank is the last token

    for token_idx in result:
        if token_idx >= len(labels):
            print(f"Warning: token_idx {token_idx} out of range for labels of length {len(labels)}")
            continue  # skip out-of-range indices
        if token_idx != blank_idx and token_idx != last_token_idx:
            decoded_text += labels[token_idx]
        last_token_idx = token_idx

    return decoded_text

# Placeholder labels - Using actual vocabulary imported from vocab.py

print("LABELS length:", len(LABELS))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not onnx_session:
        raise HTTPException(status_code=500, detail="ASR model not loaded")

    # Basic file type validation
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .wav files are supported.")

    try:
        # Read the audio file content
        audio_content = await file.read()

        # Read audio data and sample rate using soundfile
        try:
            audio_data, samplerate = sf.read(io.BytesIO(audio_content))
            if samplerate != 16000:
                 # Basic sample rate validation
                 raise HTTPException(status_code=400, detail="Invalid sample rate. Only 16kHz is supported.")

            # Check duration (approximate without full loading/processing)
            duration = len(audio_data) / samplerate
            if not (5 <= duration <= 10):
                 print(f"Warning: Audio duration ({duration:.2f}s) is outside the recommended 5-10s range.")
                 # Decide if you want to reject or warn. Warning for now.

        except Exception as e:
             raise HTTPException(status_code=400, detail=f"Error processing audio file with soundfile: {e}")

        # --- Audio Preprocessing (Mel-Spectrogram) ---
        try:
            # Convert mono to stereo if needed (soundfile reads as mono if input is mono)
            # ONNX model expects [batch_size, channels, time] or [batch_size, time] depending on model.
            # Assuming model expects [batch_size, num_features, time_frames] as seen in verify_onnx.py output
            # The compute_mel_spectrogram function returns [time_frames, num_features]
            # We need to convert it to [batch_size, num_features, time_frames]

            # Ensure audio_data is float32 as expected by ONNX model (Type: tensor(float))
            audio_data = audio_data.astype(np.float32)

            # Compute mel spectrogram (returns [time_frames, num_features])
            mel_spec = compute_mel_spectrogram(audio_data, samplerate)

            # Transpose to get [num_features, time_frames] and add batch size dimension
            # Desired ONNX input shape is ['audio_signal_dynamic_axes_1', 80, 'audio_signal_dynamic_axes_2']
            # This likely corresponds to [batch_size, num_features, time_frames]
            processed_audio = np.expand_dims(mel_spec.T, axis=0) # Add batch size of 1 -> [1, num_features, time_frames]

            # Prepare length tensor [batch_size]
            # The length input for the ONNX model is likely the number of time frames after preprocessing.
            processed_length = np.array([mel_spec.shape[0]], dtype=np.int64) # Number of time frames


        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during audio preprocessing: {e}")
        # -------------------------------------------

        # --- Perform ONNX Inference ---
        try:
            # Get input names from the ONNX session
            input_names = [input.name for input in onnx_session.get_inputs()]
            output_names = [output.name for output in onnx_session.get_outputs()]

            # Prepare input feed dictionary
            # Input names from verify_onnx.py output: audio_signal, length
            input_feed = {
                input_names[0]: processed_audio,
                input_names[1]: processed_length
            }

            # Run inference
            # output_names from verify_onnx.py output: logprobs
            outputs = onnx_session.run(output_names, input_feed)

            # Assuming the first output is the logprobs
            logprobs = outputs[0]

            print("Max index in logprobs:", np.argmax(logprobs, axis=2).max())

        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Error during ONNX inference: {e}")
        # ------------------------------

        # --- CTC Decoding ---
        try:
            # Decode the logprobs to text
            # Using the imported actual labels
            transcribed_text = ctc_greedy_decode(logprobs, LABELS)

        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Error during CTC decoding: {e}")
        # --------------------

        return {"transcribed_text": transcribed_text}

    except Exception as e:
        # Catch any remaining unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

model = onnx.load("stt_hi_conformer_ctc_medium.onnx")
optimized_model = optimizer.optimize_model(model)
onnx.save(optimized_model, "stt_hi_conformer_ctc_medium_optimized.onnx")

MODEL_PATH = "/Users/saani/Mvaak AI/stt_hi_conformer_ctc_medium.nemo"
ONNX_PATH = "/Users/saani/Mvaak AI/stt_hi_conformer_ctc_medium.onnx"

asr_model = EncDecCTCModelBPE.restore_from(MODEL_PATH)
asr_model.eval()
dummy_input = torch.randn(1, 80, 1600)  # 10s of audio at 100fps
asr_model.export(ONNX_PATH, input_example=dummy_input) 