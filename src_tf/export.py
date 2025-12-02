import tensorflow as tf
import tf2onnx
import os

def convert_to_onnx(saved_model_path, onnx_output_path, opset=13):
    """
    Converts a TensorFlow SavedModel (QAT) to ONNX with QDQ nodes.
    """
    print(f"Loading model from {saved_model_path}...")
    import tf_keras
    model = tf_keras.models.load_model(saved_model_path)
    
    print(f"Converting to ONNX (opset {opset})...")
    
    # tf2onnx.convert.from_keras automatically handles TFMOT quantization wrappers
    # and produces QuantizeLinear/DequantizeLinear nodes for opset >= 13.
    
    spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=spec, 
        opset=opset,
        output_path=onnx_output_path
    )
    
    print(f"ONNX model saved to {onnx_output_path}")
