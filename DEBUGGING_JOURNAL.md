# Debugging Journal: PyTorch QAT ONNX Export

## Issue Summary
We encountered a persistent failure when exporting a standard PyTorch QAT model to ONNX. The goal was to produce an "Explicit QDQ" graph (ops wrapped in QuantizeLinear/DequantizeLinear nodes) for the Zukimo NPU.

**Error Symptoms:**
- `UnicodeEncodeError` in the logs (masking the real error).
- `AttributeError: '...Conv2dPackedParamsBase' object has no attribute '__obj_flatten__'`
- `ValueError: Exporting a ScriptModule is not supported...`

## Root Cause Analysis
The issue stems from a regression/change in **PyTorch 2.3.0**'s ONNX exporter.
1.  **Dynamo Path**: PyTorch 2.3 defaults to a new export path (TorchDynamo) which is stricter and seemingly incompatible with the legacy `torch.quantization` path (`ScriptModule` based QAT) used for standard INT8 export.
2.  **Compatibility**: The new exporter tries to "flatten" objects that the old quantized modules don't support, causing the crash.

## Debugging Steps Taken
1.  **Standard Export (`torch.onnx.export`)**: Failed with `Conv2dPackedParamsBase` error.
2.  **`do_constant_folding=True`**: Tested to see if disabling the "Explicit" requirement enabled the export. Failed.
3.  **`torch.jit.trace`**: Attempted to trace the model before export to "freeze" the graph structure. Failed (trace captured the error).
4.  **`torch.jit.freeze`**: Attempted to freeze the traced model. Failed.
5.  **Environment Downgrade (The Fix)**:
    - We identified that PyTorch 2.0/2.1/2.2 used the legacy exporter by default for these modules.
    - We downgraded to **PyTorch 2.2.2** (compatible with the user's Python 3.12).
    - **Result**: Success. The legacy exporter correctly produced the Opset 13 Explicit QDQ graph.

## Solution Configuration
- **Python**: 3.12
- **PyTorch**: 2.2.2
- **TorchVision**: 0.17.2
- **Export Flags**: `opset_version=13`, `do_constant_folding=False`
