import onnxruntime
import os
import torch

# Replace with the actual path to your ONNX file
onnx_model_path = "/Users/saani/Mvaak AI/stt_hi_conformer_ctc_medium_simplified.onnx"

# Check if the file exists
if not os.path.exists(onnx_model_path):
    print(f"Error: ONNX model file not found at {onnx_model_path}")
else:
    try:
        # Load the ONNX model
        session = onnxruntime.InferenceSession(onnx_model_path, None)
        print(f"Successfully loaded ONNX model: {onnx_model_path}")

        # Print model inputs
        print("\nModel Inputs:")
        for input_meta in session.get_inputs():
            print(f"  Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")

        # Print model outputs
        print("\nModel Outputs:")
        for output_meta in session.get_outputs():
            print(f"  Name: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}")

        # Attempt to get opset version from metadata
        opset_version = "N/A"
        if session.get_modelmeta() and session.get_modelmeta().custom_metadata_map:
             opset_version = session.get_modelmeta().custom_metadata_map.get('opset', 'N/A')
        print(f"\nONNX Opset Version: {opset_version}")

    except Exception as e:
        print(f"Error loading or inspecting ONNX model: {e}")
