# QAT Training and QDQ ONNX Export

This project provides a modular framework for Quantization Aware Training (QAT) of TensorFlow models and exporting them to QDQ-quantized ONNX format for NPU deployment.

## Features
- **Modular Design**: Separate modules for data, models, quantization, training, and export.
- **QAT Support**: Uses `tensorflow_model_optimization` for QAT, automatically handling layer fusion and quantization simulation.
- **QDQ ONNX Export**: Exports models with explicit QuantizeLinear/DequantizeLinear nodes (Opset 13).
- **Extensible**: Easy to add custom datasets and models.

## Why TensorFlow?
This project was originally implemented in PyTorch but was migrated to TensorFlow to resolve issues with **QAT to ONNX export**.
- **PyTorch Issue**: The PyTorch QAT export often struggled to generate the correct **QDQ (Quantize-Dequantize)** nodes (`QuantizeLinear`, `DequantizeLinear`) required by specific NPU toolchains. It frequently defaulted to operator-level quantization or failed to preserve quantization parameters correctly during the ONNX conversion.
- **TensorFlow Solution**: The **TensorFlow Model Optimization Toolkit (TFMOT)** combined with `tf2onnx` provides a robust pipeline for generating standard, compliant QDQ ONNX models (Opset 13), ensuring seamless deployment to the target hardware.

## Structure
```
src_tf/
  data.py       # Data loading logic
  model.py      # Model definitions
  train.py      # Training loop and QAT logic
  export.py     # ONNX export logic
main_tf.py      # Entry point
```

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training and Exporting (CIFAR-10)
```bash
python main_tf.py --dataset cifar10 --epochs 5 --export_path resnet18_qat.onnx
```

### Custom Dataset
To use a custom dataset:
1.  Open `src_tf/data.py`.
2.  Implement a new function (e.g., `get_custom_data`) that returns `train_ds` and `val_ds` as `tf.data.Dataset` objects.
3.  Update `main_tf.py` to call your new function when the `--dataset` argument matches your dataset name.

### Adding a New Model
To add a new model:
1.  Open `src_tf/model.py`.
2.  Define your model using the **Functional API** (required for QAT compatibility).
    *   **Important**: Use `tf_keras` imports (`import tf_keras as keras`) instead of `tensorflow.keras` to ensure compatibility with `tensorflow-model-optimization`.
3.  Update `src_tf/train.py` to import and instantiate your new model.

## Visualization
You can visualize the exported ONNX model to verify the QDQ nodes (`QuantizeLinear`, `DequantizeLinear`).

### Using Netron
1.  **Web**: Go to [netron.app](https://netron.app) and open `outputs_tf/model_qat.onnx`.
2.  **Local**: Install and run Netron:
    ```bash
    pip install netron
    netron outputs_tf/model_qat.onnx
    ```

## QAT Details
The project uses **TensorFlow Model Optimization Toolkit (TFMOT)** for QAT.
1. **Quantization**: Uses `tfmot.quantization.keras.quantize_model` to simulate quantization during training (FakeQuant nodes).
2. **Export**: The model is exported to ONNX using `tf2onnx` with `opset=13`, producing standard QDQ nodes (`QuantizeLinear`, `DequantizeLinear`) compatible with NPU toolchains.

## References
- [NVIDIA TensorFlow Quantization Toolkit Documentation](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/tensorflow-quantization-toolkit/docs/index.html#:~:text=Getting%20Started:%20End%20to%20End,-NVIDIA®%20TensorFlow&text=Initially%2C%20the%20network%20is%20trained,called%20“fine%2Dtuning”.&text=Train%20a%20simple%20network%20on,it%20as%20the%20QAT%20model.)
