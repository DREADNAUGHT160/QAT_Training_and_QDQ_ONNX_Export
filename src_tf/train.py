import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os
from .model import ResNet18

def train_qat_model(train_ds, val_ds, epochs=5, output_dir='outputs_tf'):
    """
    Trains a ResNet18 model with Quantization Aware Training (QAT).
    """
    # 1. Build the base model
    # CIFAR-10 input shape is (32, 32, 3)
    model = ResNet18(input_shape=(32, 32, 3), num_classes=10)
    
    # 2. Apply QAT
    # This wraps layers with quantization logic (FakeQuant nodes)
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)
    
    # 3. Compile
    # Using Adam usually works well for TF, but SGD is standard for ResNet.
    # We'll use SGD with momentum to match the PyTorch setup.
    import tf_keras
    optimizer = tf_keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    
    q_aware_model.compile(
        optimizer=optimizer,
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    q_aware_model.summary()
    
    # 4. Train
    print("Starting QAT training...")
    q_aware_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds
    )
    
    # 5. Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, 'qat_model')
    q_aware_model.save(save_path)
    print(f"QAT model saved to {save_path}")
    
    return save_path
