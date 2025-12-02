import argparse
import os
from src_tf.data import get_cifar10_data
from src_tf.train import train_qat_model
from src_tf.export import convert_to_onnx

def main():
    parser = argparse.ArgumentParser(description='TensorFlow QAT Training and ONNX Export')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='outputs_tf', help='Output directory')
    
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading data...")
    train_ds, val_ds = get_cifar10_data(batch_size=args.batch_size)
    
    # 2. Train QAT Model
    saved_model_path = train_qat_model(train_ds, val_ds, epochs=args.epochs, output_dir=args.output_dir)
    
    # 3. Export to ONNX
    onnx_path = os.path.join(args.output_dir, 'model_qat.onnx')
    convert_to_onnx(saved_model_path, onnx_path)
    
    print("Pipeline complete!")

if __name__ == '__main__':
    main()
