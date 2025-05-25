from nemo.collections.asr.models import EncDecCTCModelBPE

# Path to your .nemo model
MODEL_PATH = "/Users/saani/Mvaak AI/stt_hi_conformer_ctc_medium.nemo"
ONNX_PATH = "/Users/saani/Mvaak AI/stt_hi_conformer_ctc_medium_simplified.onnx"

# Load the model
asr_model = EncDecCTCModelBPE.restore_from(MODEL_PATH)
asr_model.eval()

# Export to ONNX using the official NeMo .export() method
asr_model.export(ONNX_PATH)
print(f"Exported ONNX model to {ONNX_PATH}")
