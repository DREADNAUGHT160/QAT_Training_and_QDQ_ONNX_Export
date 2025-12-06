import torch
import torch.onnx
from model import QATModel
import os

def export_model():
    device = torch.device('cpu') # Export usually done on CPU
    
    # 1. Initialize and Fuse
    model = QATModel(num_classes=10)
    model.eval()
    model.fuse_model()
    model.train() # prepare_qat requires training mode
    
    # 2. Prepare (Must match training setup!)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 3. Load Weights
    load_path = "qat_model.pth"
    if not os.path.exists(load_path):
        print(f"Error: {load_path} not found. Run train.py first.")
        return

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval() # Important for some layers, though convert handles QAT switch
    
    print("QAT Model loaded.")

    # DEBUG: Export Float Model first to check toolchain
    try:
        dummy_input = torch.randn(1, 3, 32, 32)
        torch.onnx.export(model, dummy_input, "model_float.onnx", opset_version=13)
        print("Float model exported successfully.")
    except Exception as e:
        print(f"Float model export failed: {e}")

    # 4. Convert to Quantized Model
    # This replaces FakeQuant with actual Quantized layers (or QuantizeLinear nodes in ONNX)
    torch.quantization.convert(model, inplace=True)
    print("Model converted to quantized version.")
    
    # 5. Export to ONNX
    # CRITICAL: do_constant_folding=False
    dummy_input = torch.randn(1, 3, 32, 32)
    output_path = "model_qdq.onnx"
    
    # 5. Export to ONNX
    # CRITICAL: do_constant_folding=False
    dummy_input = torch.randn(1, 3, 32, 32)
    output_path = "model_qdq.onnx"
    
    # Direct export works in PyTorch 2.0 for QAT
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13,
        do_constant_folding=False, # <--- THE KEY REQUIREMENT
        input_names=['input'],
        output_names=['output']
    )
    
    print(f"Model exported to {output_path} with Explicit QDQ format.")

if __name__ == "__main__":
    export_model()
