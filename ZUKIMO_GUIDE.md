# Zukimo NPU: PyTorch QAT to ONNX Pipeline Guide

This document provides a complete, production-ready pipeline for exporting a fully quantized PyTorch QAT model into ONNX and compiling it with the Zukimo NPU toolchain.

---

## Section 1: Conceptual Explanation

### 1. What is Fake Quantization?
Fake quantization is a technique used during training to simulate the effects of low-precision (e.g., INT8) arithmetic without actually converting the weights and activations to integers.
- **Q (Quantize)**: Converts a float value to an integer index.
- **DQ (Dequantize)**: Converts the integer index back to a float approximation.
- **Operation**: $x_{out} = DQ(Q(x_{in}))$
- **Purpose**: This introduces "quantization noise" into the forward pass, forcing the model to learn weights that are robust to this noise.

### 2. Why PyTorch QAT Inserts Q/DQ Nodes
PyTorch's QAT workflow inserts `QuantStub` and `DeQuantStub` modules, along with "fake quantize" observers on weights and activations.
- **During Training**: These modules track min/max ranges and simulate precision loss.
- **During Conversion**: PyTorch uses these observed ranges to calculate **Scale** and **Zero-Point** for every tensor.

### 3. QAT-Prepared vs. QAT-Converted
- **QAT-Prepared Model**: A float32 model with "observer" modules attached. It runs in float32 but collects statistics.
- **QAT-Converted Model**: A model where layers are replaced by their quantized counterparts (e.g., `nn.Conv2d` $\to$ `nn.quantized.Conv2d`). The weights are stored as INT8 (or uint8), and operations are integer-based.

### 4. The ONNX Export Nuance
When you export a *converted* PyTorch quantized model to ONNX:
- **Standard Export**: Often produces a graph with `QLinearConv` (operator-oriented).
- **Zukimo Requirement**: Zukimo prefers **Tensor-Oriented** quantization (Opset 13+). This means the graph should look like:
  `Input (float) -> QuantizeLinear -> (Int8 Data) -> ConvInteger -> (Int32 Data) -> DequantizeLinear -> Output (float)`
  
  *However*, for many NPU compilers (including Zukimo), the most robust format is often **Explicit QDQ**:
  `Input -> Q -> DQ -> Conv (Float) -> Q -> DQ -> Output`
  
  **Wait, why Float Conv?**
  If we export with `do_constant_folding=False` and `opset_version=13`, PyTorch/ONNX often represents the graph as "Fake Quantized" nodes (QuantizeLinear -> DequantizeLinear) wrapping standard float operators. The *Compiler* (Zukimo) then fuses these $Q \to DQ \to Op \to Q \to DQ$ patterns into a single hardware-accelerated INT8 operation.

### 5. Zukimo YAML Requirements
The Zukimo toolchain needs a YAML configuration to map the ONNX tensors to hardware memory buffers.
- **IQ (Input Quantization)**: Scale/Zero-point for the input tensor.
- **OQ (Output Quantization)**: Scale/Zero-point for the output tensor.
- **WQ (Weight Quantization)**: Scale/Zero-point for weights.
- **BQ (Bias Quantization)**: Scale/Zero-point for biases.

**Critical Error**: `KeyError: 'OQ'` usually means the compiler cannot determine the quantization parameters for a specific node's output, often because a `QuantizeLinear` node is missing or disconnected in the ONNX graph.

---

## Section 2: Full Working PyTorch Pipeline

### 1. Dependencies
```bash
pip install torch torchvision onnx netron numpy
```

### 2. Code Implementation

See `src/model.py`, `src/train.py`, and `src/export.py` in the codebase for the full implementation.

#### Key Steps in Code:
1.  **Model Definition**: Use `QuantStub` at input and `DeQuantStub` at output.
2.  **Fusion**: Fuse `Conv2d + BatchNorm2d + ReLU` into a single module. This is critical because the NPU executes them as one block.
3.  **QConfig**: Use `torch.quantization.get_default_qat_qconfig('fbgemm')`.
4.  **Prepare**: `torch.quantization.prepare_qat(model, inplace=True)`.
5.  **Convert**: `torch.quantization.convert(model, inplace=True)`.
6.  **Export**:
    ```python
    torch.onnx.export(
        model,
        dummy_input,
        "model_int8.onnx",
        opset_version=13,
        do_constant_folding=False,  # CRITICAL for preserving Q/DQ info
        input_names=['input'],
        output_names=['output']
    )
    ```

---

## Section 3: Validation in Netron

1.  **Open** `model_int8.onnx` in [Netron](https://netron.app).
2.  **Check Input**: Ensure the input feeds into a `QuantizeLinear` node.
3.  **Check Weights**: Click on a Convolution node. The weights should be inputs from a `DequantizeLinear` node (if explicit QDQ) or directly `Int8` (if QLinearConv).
    - *Preferred for Zukimo*: Explicit `QuantizeLinear` -> `DequantizeLinear` wrapping the weights.
4.  **Check Attributes**: Ensure `x_scale` and `x_zero_point` are present and correct (not all zeros).

---

## Section 4: Zukimo Toolchain Pipeline

Assuming you have the `zukimo-cli` installed, the workflow is:

### 1. Copy Input
```bash
zukimo-cli copy_input --input model_int8.onnx --output workspace/
```

### 2. Convert Model (Optional/Internal)
Some versions require an intermediate conversion.
```bash
zukimo-cli convert_model --model workspace/model_int8.onnx
```

### 3. Generate RISC-V Config (The YAML Step)
This is where the magic happens. The tool reads the ONNX QDQ nodes and extracts `IQ`, `OQ`, `WQ`.
```bash
zukimo-cli generate_riscv --model workspace/model_int8.onnx --config zukimo.yaml
```

**YAML Snippet Example:**
```yaml
layers:
  - name: conv1
    type: Conv2D
    IQ: {scale: 0.0156, zeropoint: 128}
    WQ: {scale: 0.0012, zeropoint: 0}
    OQ: {scale: 0.0312, zeropoint: 128}
```

### 4. Compile & Bundle
```bash
zukimo-cli compile_riscv --config zukimo.yaml --output binary.bin
zukimo-cli bundle_files --binary binary.bin --output bundle.pkg
```

---

## Section 5: Debugging Guide

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| **KeyError: 'OQ'** | The tool cannot find output quantization params. | Ensure the ONNX model has a `QuantizeLinear` node *immediately* following the layer in question. Check `do_constant_folding=False`. |
| **Float32 ONNX** | Exported before `convert()` or wrong opset. | Call `torch.quantization.convert(model)` *before* export. Use `opset_version=13`. |
| **Missing Quant Params** | Observers didn't run or stats are NaN. | Ensure you ran calibration (forward passes) on representative data. Check for NaNs in input. |
| **Unsupported Ops** | Model uses layers not supported by Zukimo. | Stick to standard `Conv2d`, `Linear`, `ReLU`, `MaxPool`. Avoid dynamic slicing or complex reshaping. |

---

## Section 6: Best Practices

1.  **Always Convert**: Never export the QAT-prepared model directly. Always `convert()` to the quantized version first.
2.  **No Constant Folding (`do_constant_folding=False`)**:
    - **Why is this critical?** By default, ONNX export tries to optimize the graph by pre-calculating operations that have constant inputs.
    - **The Problem**: In a quantized model, the "Scale" and "Zero-Point" are often constants. If constant folding is enabled, the exporter might "bake" the quantization into the weights directly (e.g., `Weight_Float = (Weight_Int - ZP) * Scale`).
    - **The Result**: You get a graph with raw float weights and NO `QuantizeLinear`/`DequantizeLinear` nodes wrapping them. The Zukimo compiler looks for these specific Q/DQ nodes to identify that a layer is quantized. If they are missing (because they were optimized away), the compiler treats the layer as standard Float32.
    - **The Fix**: Setting `do_constant_folding=False` forces the exporter to keep the `QuantizeLinear` and `DequantizeLinear` nodes explicitly in the graph, ensuring the compiler can "see" the quantization parameters.
3.  **Opset 13**: The gold standard for modern quantization support in ONNX.
4.  **Simple Architectures**: Start with a simple CNN to validate the toolchain before moving to ResNets or Transformers.
5.  **Sanity Check**: If the accuracy drops to random chance (10% for CIFAR-10), your quantization ranges are likely broken. Check `observer` stats in PyTorch.
