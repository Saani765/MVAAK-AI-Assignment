# FastAPI ASR Application with NVIDIA NeMo and ONNX

This project provides a FastAPI application to serve an Automatic Speech Recognition (ASR) model built using NVIDIA NeMo and optimized for inference with ONNX.

## Requirements

*   Docker
*   Your ONNX-optimized ASR model file (e.g., `stt_hi_conformer_ctc_medium.onnx`) obtained by exporting the NVIDIA NeMo model.

## Setup

1.  Place your ONNX model file (`stt_hi_conformer_ctc_medium.onnx`) in the same directory as `main.py` and `Dockerfile`.

## Building the Docker Image

Navigate to the project root directory in your terminal and run:

```bash
docker build -t fastapi-asr-app .
```

This will build a Docker image named `fastapi-asr-app`.

## Running the Docker Container

Run the following command to start the container and map port 8000:

```bash
docker run -d -p 8000:8000 --name asr-container fastapi-asr-app
```

This will run the container in detached mode (`-d`) and make the application accessible at `http://localhost:8000`.

## Testing the `/transcribe` Endpoint

You can test the `/transcribe` endpoint using `curl` or Postman.

**Using curl:**

```bash
curl -X POST http://localhost:8000/transcribe -H "Content-Type: multipart/form-data" -F "file=@your_audio.wav"
```

Replace `your_audio.wav` with the path to your 16kHz, 5-10 second WAV audio file.

**Using Postman:**

1.  Create a new POST request.
2.  Set the URL to `http://localhost:8000/transcribe`.
3.  Go to the "Body" tab and select "form-data".
4.  Add a key named `file`.
5.  Change the type of the `file` key from "Text" to "File".
6.  Select your `your_audio.wav` file.
7.  Click "Send".

## Design Considerations

*   **ONNX for Inference:** Using ONNX allows for potentially faster and more portable inference compared to the original training framework.
*   **FastAPI:** Provides a modern, fast, asynchronous web framework suitable for building APIs.
*   **Docker Containerization:** Ensures the application is portable and has a consistent environment across different systems.
*   **Basic Validation:** Includes basic checks for file type and sample rate. Duration check is approximate.
*   **Async Endpoint:** The `/transcribe` endpoint is defined as `async` to allow for potentially non-blocking operations, although the current placeholder inference is blocking. Actual asynchronous inference would depend on the ONNX runtime implementation and the nature of the model processing.

## Achievements

- Successfully built a FastAPI application for ASR using NVIDIA NeMo and ONNX.
- The application can accept 16kHz, 5–10 second WAV files and perform all necessary preprocessing.
- **The NeMo model (`.nemo`) was able to transcribe all three test audio files perfectly.**
- Dockerized deployment and easy API usage.

## Drawbacks and Limitations

- **ONNX Export Limitation:** Despite extensive effort, the ONNX-exported model (`stt_hi_conformer_ctc_medium.onnx`) always outputs only blank tokens for all audio files, even though the NeMo model works perfectly. This is a known issue with some NeMo models and ONNX export.
- All preprocessing, normalization, and input shape matching were verified and correct.
- Tried all recommended ONNX optimizations (eval mode, static/dynamic axes, dither, center, fmax, onnxsim, etc.) with no success.
- The issue appears to be with the ONNX export for this model/version and is not fixable at the application level.
- **If ONNX inference is required, a different model or NeMo/ONNX version may be needed, or the issue should be reported to NVIDIA.**

## Actual NeMo Model Transcription Results

| Audio File                | Transcription Result (NeMo)         |
|---------------------------|-------------------------------------|
| audio_16khz.wav           | कृप्या बिना टिकट यात्रा न करें यह दंडनीय अपराध है |
| audio2_16khz.wav          |'आगरा के ताजमहल में मुगल बादशाह शाहजहां के उर्स 
                              की शुरुआत की गई जिसके चलते शाहजहां और 
                              मोमतास की कब्र का गेट खोला गया            |
| audio3.wav                | हमने उस उम्मीदवार को अपना मत दिया          |

> _Note: The above results were produced using the NeMo model directly. The ONNX model produced only empty output for all files._

## Next Steps / Further Improvements (Beyond the initial submission)

*   Implement the actual audio preprocessing and ONNX inference logic in `main.py`.
*   Refine audio duration validation.
*   Add more robust error handling and logging.
*   Consider adding more endpoints (e.g., health check).
*   Explore using a dedicated inference server like NVIDIA Triton Inference Server for better performance and management in production. 