# Setup and Usage Guide

This guide is for developers or users who need to generate the **Explicit QDQ ONNX model** for the Zukimo NPU using this repository.

## ⚠️ Critical Requirement
**Do NOT simply run `pip install torch`**.
This project requires **PyTorch 2.2.2** (or 2.0.1).
- **Newer versions (PyTorch 2.3+)**: Have a regression that breaks the QAT export.
- **Older versions**: May not support Python 3.12.

## 1. Environment Setup

### Option A: Using Pip (Recommended)
1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd QAT_model
    ```
2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *This will automatically install correct versions: `torch==2.2.2`, `torchvision==0.17.2`.*

### Option B: Using Conda
```bash
conda create -n qat_env python=3.12
conda activate qat_env
pip install -r requirements.txt
```

---

## 2. Running the Pipeline

The pipeline consists of two steps: **Training** (to create the quantized model) and **Exporting** (to convert it to ONNX).

### Step 1: Train and Calibrate
Run the training script. This script:
- Fuses model layers (`Conv + BN + ReLU`).
- Inserts "Fake Quantization" nodes.
- Calibrates the model on CIFAR-10 data.
- Saves the checkpoint to `qat_model.pth`.

```bash
python src_torch/train.py
```
*Expected Output*: `Model prepared... Starting QAT... QAT-Prepared model saved to qat_model.pth`.

### Step 2: Export to ONNX
Run the export script. This transforms the PyTorch QAT model into an "Explicit QDQ" ONNX graph (Opset 13).

```bash
python src_torch/export.py
```
*Expected Output*: `Model converted... Model exported to model_qdq.onnx`.

> [!NOTE]
> You may see a warning: `Float model export failed`. **Ignore this.**
> This is a debug check failing on the quantization nodes. It does **not** affect the final `model_qdq.onnx`.

---

## 3. Verifying the Output

You should now have a file named `model_qdq.onnx` in the root directory.

To verify it is correct for Zukimo:
1.  Open `model_qdq.onnx` in [Netron](https://netron.app).
2.  Locate any Convolution layer.
3.  **Check**: The inputs to the Convolution (both data and weights) should come from `DequantizeLinear` nodes.
4.  **Check**: The output of the Convolution should go into a `QuantizeLinear` node.

This "sandwich" structure (`Q -> DQ -> Op -> Q -> DQ`) is what allows the Zukimo compiler to generate the necessary `IQ`, `OQ`, and `WQ` parameters.

---

## Troubleshooting

### "AttributeError: ... Conv2dPackedParamsBase"
**Cause**: You are likely running PyTorch 2.3 or newer.
**Fix**: Uninstall torch and reinstall the correct version:
```bash
pip uninstall torch torchvision
pip install torch==2.2.2 torchvision==0.17.2
```
