import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import QATModel
from data_loader import get_dataloaders

def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print(f'[Step {i + 1}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = QATModel(num_classes=10).to(device)
    
    # 2. Fuse Modules (Critical for NPU efficiency & QAT stability)
    # Fusion requires eval mode in some PyTorch versions
    model.eval()
    model.fuse_model()
    # Switch back to train because prepare_qat requires training mode
    model.train()
    
    # 3. Define QConfig
    # 'fbgemm' is standard for x86. NPU might prefer generic, but fbgemm works for export.
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # 4. Prepare for QAT
    # This inserts observers and fake_quant modules
    torch.quantization.prepare_qat(model, inplace=True)
    
    print("Model prepared for QAT. Structure:")
    # print(model) # Too verbose?

    # 5. Pipeline Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainloader, testloader = get_dataloaders(batch_size=32)

    # 6. Train (QAT)
    # In a real scenario, you train for many epochs. 
    # For this demonstration/template, we do fewer.
    epochs = 1
    print(f"Starting QAT for {epochs} epoch(s)...")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_one_epoch(model, criterion, optimizer, trainloader, device)
        acc = evaluate(model, testloader, device)
        print(f"Validation Accuracy: {acc:.2f}%")

    # 7. Save QAT Model (still with FakeQuants, not converted yet)
    # We save this so export.py can load it, convert it, and export it.
    save_path = "qat_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"QAT-Prepared model saved to {save_path}")

if __name__ == "__main__":
    main()
