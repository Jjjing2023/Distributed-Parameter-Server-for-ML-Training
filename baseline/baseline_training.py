import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Data loading and preprocessing
def load_cifar100():
    """Load and preprocess CIFAR-100 dataset"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Download CIFAR-100
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    return trainset, testset

# Simple ResNet-18 implementation for CIFAR-100
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        
        # Final classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class TrainingMetrics:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.epoch_times = []
        self.total_time = 0
        
    def add_epoch(self, train_loss, train_acc, test_acc, epoch_time):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        self.epoch_times.append(epoch_time)
        self.total_time += epoch_time
    
    def plot_results(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curve
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train')
        ax2.plot(self.test_accuracies, label='Test')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        # Training time per epoch
        ax3.plot(self.epoch_times)
        ax3.set_title('Time per Epoch')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        
        # Summary stats
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        final_test_acc = self.test_accuracies[-1]
        ax4.text(0.1, 0.8, f'Final Test Accuracy: {final_test_acc:.2f}%', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f'Total Training Time: {self.total_time/60:.1f} minutes', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f'Avg Time/Epoch: {avg_epoch_time:.1f} seconds', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.2, f'Epochs Completed: {len(self.epoch_times)}', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Training Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('baseline_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def train_epoch(model, trainloader, criterion, optimizer, device, training_start_time):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:  # Progress updates every 10 batches
            elapsed_epoch = time.time() - epoch_start
            elapsed_total = time.time() - training_start_time
            print(f'Batch {batch_idx}/{len(trainloader)}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%, Elapsed: {elapsed_epoch/60:.1f}min this epoch, '
                  f'{elapsed_total/60:.1f}min total')
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def test_epoch(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_acc = 100. * correct / total
    return test_acc

def main():
    # Configuration
    BATCH_SIZE = 128
    EPOCHS = 3  # Just for timing estimation
    LEARNING_RATE = 0.1
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    trainset, testset = load_cifar100()
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Dataset loaded - Train: {len(trainset)} samples, Test: {len(testset)} samples")
    
    # Initialize model
    model = ResNet18(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {total_params:,} parameters')
    
    # Training loop
    metrics = TrainingMetrics()
    print("\nStarting baseline training...")
    training_start_time = time.time()  # Track total training start time
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, training_start_time)
        test_acc = test_epoch(model, testloader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start_time
        metrics.add_epoch(train_loss, train_acc, test_acc, epoch_time)
        
        print(f'Epoch {epoch+1} completed in {epoch_time:.1f}s (Total elapsed: {total_elapsed/60:.1f}min)')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    print(f'\nBaseline training completed!')
    print(f'Total time: {metrics.total_time/60:.1f} minutes')
    print(f'Final test accuracy: {metrics.test_accuracies[-1]:.2f}%')
    
    # Generate results visualization
    metrics.plot_results()
    
    # Save baseline results for your report
    results = {
        'final_accuracy': metrics.test_accuracies[-1],
        'total_time_minutes': metrics.total_time / 60,
        'avg_epoch_time': sum(metrics.epoch_times) / len(metrics.epoch_times),
        'model_parameters': total_params,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS
    }
    
    print("\nBaseline Results Summary:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main()