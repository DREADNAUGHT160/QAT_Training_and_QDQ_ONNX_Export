# PyTorch QAT to ONNX (Explicit QDQ) Pipeline

This project implements a **Quantization-Aware Training (QAT)** pipeline using **PyTorch 2.2.2**. It is designed to export models in the specific "Explicit QDQ" ONNX format required by the Zukimo NPU toolchain.

## 🚀 Key Features
- **Explicit QDQ Export**: Generates ONNX graphs where `QuantizeLinear` and `DequantizeLinear` nodes are preserved (not folded), enabling the NPU compiler to extract quantization parameters (`IQ`, `OQ`, `WQ`).
- **Fused Architecture**: Implements `Conv + BatchNorm + ReLU` fusion for hardware efficiency.
- **Custom Model**: A lightweight 3-layer CNN optimized for CIFAR-10.

## ❓ Why Explicit Quantization?
Standard ONNX export often "folds" quantization parameters (Scale/ZeroPoint) directly into the weights to simplify the graph for CPU/GPU inference.
**However, the Zukimo NPU Compiler requires the Raw Quantization Nodes.**

It needs to see:
- `QuantizeLinear` (The "Q" step)
- `DequantizeLinear` (The "DQ" step)

By keeping these nodes **Explicit** in the graph (using `do_constant_folding=False`), the compiler can extract the exact `IQ` (Input Quantization), `OQ` (Output Quantization), and `WQ` (Weight Quantization) parameters needed to configure the hardware accelerators. Without them, the compiler sees "Float32" operations and fails.

---

## 🏗️ Model Architecture
The model (`src_torch/model.py`) is a custom VGG-style CNN designed for 32x32 inputs (CIFAR-10 classification).

| Layer Block | Components | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| **Input** | `QuantStub` | `(3, 32, 32)` | Converts Float $\to$ Int8 |
| **Block 1** | `Conv2d` (3$\to$32, 3x3) + `BN` + `ReLU` + `MaxPool` | `(32, 16, 16)` | Fused OP |
| **Block 2** | `Conv2d` (32$\to$64, 3x3) + `BN` + `ReLU` + `MaxPool` | `(64, 8, 8)` | Fused OP |
| **Block 3** | `Conv2d` (64$\to$64, 3x3) + `BN` + `ReLU` | `(64, 8, 8)` | Fused OP |
| **Head** | `Flatten` + `Linear` (4096$\to$10) | `(10)` | Fully Connected |
| **Output** | `DeQuantStub` | `(10)` | Converts Int8 $\to$ Float |

---

## 🛠️ Usage

### 1. Requirements
**Crucial**: You must use **PyTorch 2.2.2** (or 2.0.1) for the export to work correctly. Newer versions (2.3+) currently fail to export standard QAT graphs.
```bash
pip install -r requirements.txt
```

### 2. Training (QAT)
Trains the model, performs fake quantization, fuses layers, and saves the checkpoint.
```bash
python src_torch/train.py
```
**Output**: `qat_model.pth`

### 3. Export (ONNX)
Converts the QAT model to a quantized version and exports it to ONNX using `opset 13`.
```bash
python src_torch/export.py
```
**Output**: `model_qdq.onnx`

---

## 📦 Artifacts
- **`model_qdq.onnx`**: The final artifact ready for the Zukimo NPU compiler.
