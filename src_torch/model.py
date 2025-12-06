import torch
import torch.nn as nn
import torch.quantization

class QATModel(nn.Module):
    def __init__(self, num_classes=10):
        super(QATModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        
        # Layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        # Fully Connected
        # Input size calculation: 32x32 -> 16x16 -> 8x8. 64 channels * 8 * 8 = 4096
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # Fuse Conv+BN+ReLU modules
        # Note: PyTorch requires a list of module names to fuse
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'], 
                                               ['conv2', 'bn2', 'relu2'],
                                               ['conv3', 'bn3', 'relu3']], inplace=True)

if __name__ == '__main__':
    # Sanity check
    model = QATModel()
    print(model)
    model.fuse_model()
    print("Fused Model:")
    print(model)
