import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(batch_size=32, root='./data'):
    """
    Returns train and test dataloaders for CIFAR10.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure size matches model
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                               download=True, transform=transform)
    except Exception as e:
        print(f"Could not download CIFAR10: {e}. Using FakeData.")
        trainset = torchvision.datasets.FakeData(size=1000, image_size=(3, 32, 32),
                                                 num_classes=10, transform=transform)
        testset = torchvision.datasets.FakeData(size=100, image_size=(3, 32, 32),
                                                num_classes=10, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    
    return trainloader, testloader
