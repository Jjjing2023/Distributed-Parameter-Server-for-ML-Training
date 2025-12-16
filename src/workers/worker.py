import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grpc
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
from torch.utils.data import DataLoader, Subset
import threading

from communication import ps_pb2
from communication import ps_pb2_grpc

# --- ResNet-18 Model (matching server) ---
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
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        
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

class DistributedWorker:
    def __init__(self, server_address='localhost:8000', worker_name='worker-node'):
        self.server_address = server_address
        self.worker_name = worker_name
        self.worker_id = None
        self.total_workers = None
        
        # Training configuration
        self.batch_size = 512  # Match baseline
        self.learning_rate = 0.1  # Match baseline
        self.num_epochs = 3  # Match baseline (shortened for testing)
        self.local_steps_per_sync = 5  # Sync after every batch
        
        # Model and data
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        
        # gRPC setup
        self.channel = None
        self.stub = None
        
        # Training tracking
        self.local_step_counter = 0
        self.global_step_cache = 0
        
        # Performance metrics
        self.training_start_time = None
        self.epoch_times = []
        self.accuracies = []
        self.training_active = True

    def health_check_loop(self):  # Add this entire method
        while self.training_active:
            try:
                self.stub.FetchParameters(ps_pb2.FetchRequest(worker_id=self.worker_id))
                time.sleep(30)
            except:
                print("Connection lost, attempting reconnect...")
                self.connect_to_server()

    def run_training(self):
        # Start health check thread
        health_thread = threading.Thread(target=self.health_check_loop)
        health_thread.daemon = True
        health_thread.start()
        
        
    def setup_model(self):
        """Initialize ResNet-18 model matching server structure"""
        print("Setting up ResNet-18 model for CIFAR-100...")
        
        self.model = ResNet18(num_classes=100)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized with {total_params:,} parameters")
    
    def setup_data(self):
        """Setup CIFAR-100 dataset with worker-specific data partition"""
        print("Loading CIFAR-100 dataset...")
        
        # CIFAR-100 transforms (matching baseline)
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
        
        # Load datasets
        full_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        # Partition training data among workers
        total_samples = len(full_dataset)
        samples_per_worker = total_samples // self.total_workers
        start_idx = self.worker_id * samples_per_worker
        
        if self.worker_id == self.total_workers - 1:
            # Last worker gets remaining samples
            end_idx = total_samples
        else:
            end_idx = start_idx + samples_per_worker
        
        # Create worker-specific subset
        worker_indices = list(range(start_idx, end_idx))
        worker_dataset = Subset(full_dataset, worker_indices)
        
        # Create data loaders
        self.train_loader = DataLoader(
            worker_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        print(f"Worker {self.worker_id} assigned {len(worker_dataset)} training samples (indices {start_idx}-{end_idx-1})")
        print(f"Training batches per epoch: {len(self.train_loader)}")
    
    def connect_to_server(self):
        """Establish connection to parameter server"""
        print(f"Connecting to server at {self.server_address}...")
        
        options = [
            ('grpc.max_receive_message_length', 500 * 1024 * 1024),  # 500 MB
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True)
        ]
        
        self.channel = grpc.insecure_channel(self.server_address, options=options)
        self.stub = ps_pb2_grpc.ParameterServerStub(self.channel)
        
        # Register with server
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.stub.RegisterWorker(
                    ps_pb2.RegisterRequest(worker_ip=self.worker_name)
                )
                self.worker_id = response.worker_id
                self.total_workers = response.total_workers
                print(f"Registered as Worker {self.worker_id} (Total workers: {self.total_workers})")
                return
                
            except grpc.RpcError as e:
                print(f"Registration attempt {attempt + 1} failed: {e.code()} - {e.details()}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def fetch_parameters(self):
        """Fetch latest parameters from server"""
        try:
            response = self.stub.FetchParameters(
                ps_pb2.FetchRequest(worker_id=self.worker_id)
            )
            
            # Deserialize parameters
            server_params = pickle.loads(response.parameters)
            global_step = response.global_step
            
            # Update global step cache
            self.global_step_cache = global_step
            
            # Convert numpy arrays back to tensors and load into model
            model_state_dict = {}
            for name, param_array in server_params.items():
                model_state_dict[name] = torch.tensor(param_array)
            
            self.model.load_state_dict(model_state_dict)
            print(f"  > Fetched parameters from global step {global_step}")
            
            return global_step
            
        except grpc.RpcError as e:
            print(f"Failed to fetch parameters: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            print(f"Error loading parameters: {e}")
            raise
    
    def compress_gradients(self, gradients):  # Add this new method
        compressed = {}
        for name, grad in gradients.items():
            compressed[name] = grad.astype(np.float16)
        return compressed

    def push_gradients(self):
        """Push computed gradients to server"""
        try:
            # Extract gradients from model
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.cpu().numpy()
                else:
                    print(f"Warning: No gradient for parameter {name}")
            
            # Check original size
            original_bytes = pickle.dumps(gradients)
            original_size = len(original_bytes)

            # Compress gradients
            gradients = self.compress_gradients(gradients)

            # Check compressed size and serialize once
            gradients_bytes = pickle.dumps(gradients)
            compressed_size = len(gradients_bytes)

            print(f"  > Gradient size: {original_size/1024/1024:.1f}MB → {compressed_size/1024/1024:.1f}MB ({compressed_size/original_size:.2%})")

            # Send to server (using the already serialized gradients_bytes)
            response = self.stub.PushGradrients(
                ps_pb2.PushRequest(
                    worker_id=self.worker_id,
                    gradients=gradients_bytes,  # Already serialized above
                    local_step=self.global_step_cache
                )
            )
            
            print(f"  > Pushed gradients (Size: {len(gradients_bytes)} bytes, Success: {response.received})")
            return response.received
            
        except grpc.RpcError as e:
            print(f"Failed to push gradients: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            print(f"Error pushing gradients: {e}")
            raise

    def evaluate_model(self):
        """Evaluate model accuracy on test set"""
        print("Evaluating model accuracy...")
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        print(f"  > Test Accuracy: {accuracy:.2f}%")
        self.accuracies.append(accuracy)
        return accuracy
    
    def train_local_batch(self, data, targets):
        """Train on a single batch"""
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        
        # Backward pass (compute gradients, don't step)
        loss.backward()
        
        self.local_step_counter += 1
        return loss.item()
    
    def run_training(self):
        """Main training loop with parameter server synchronization"""
        print(f"\n--- Starting distributed training for {self.num_epochs} epochs ---")
        print(f"Worker {self.worker_id}: {len(self.train_loader)} batches per epoch")
        
        self.training_start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                epoch_start = time.time()
                print(f"\nEpoch {epoch + 1}/{self.num_epochs} - Worker {self.worker_id}")
                
                epoch_loss = 0.0
                batches_processed = 0
                
                for batch_idx, (data, targets) in enumerate(self.train_loader):
                    # 1. Fetch latest parameters from server
                    if batch_idx % self.local_steps_per_sync == 0:
                        self.fetch_parameters()
                    
                    # 2. Train on local batch
                    batch_loss = self.train_local_batch(data, targets)
                    epoch_loss += batch_loss
                    batches_processed += 1
                    
                    # 3. Push gradients to server
                    if batch_idx % self.local_steps_per_sync == 0:
                        self.push_gradients()
                    
                    # Progress reporting
                    if batch_idx % 50 == 0:
                        elapsed = time.time() - epoch_start
                        print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {batch_loss:.4f}, "
                              f"Elapsed: {elapsed/60:.1f}min")
                
                # Epoch completion
                epoch_time = time.time() - epoch_start
                self.epoch_times.append(epoch_time)
                avg_loss = epoch_loss / batches_processed
                
                print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s, Avg Loss: {avg_loss:.4f}")
                
                # Evaluate at end of each epoch
                if epoch % 1 == 0:  # Evaluate every epoch
                    self.evaluate_model()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
            raise
        finally:
            self.cleanup()
            self.print_worker_statistics()
    
    def print_worker_statistics(self):
        """Print worker-specific training statistics"""
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
            final_accuracy = self.accuracies[-1] if self.accuracies else 0

            print(f"\n--- Worker {self.worker_id} Statistics ---")
            print(f"Total training time: {total_time:.1f} seconds")
            print(f"Average epoch time: {avg_epoch_time:.1f} seconds")
            print(f"Final accuracy: {final_accuracy:.2f}%")
            print(f"Local steps completed: {self.local_step_counter}")
            print("--- End Statistics ---")

            # Output structured JSON metrics for easy parsing
            import json
            metrics = {
                "type": "WORKER_FINAL_METRICS",
                "worker_id": self.worker_id,
                "total_workers": self.total_workers,
                "total_training_time_seconds": round(total_time, 2),
                "average_epoch_time_seconds": round(avg_epoch_time, 2),
                "epoch_times_seconds": [round(t, 2) for t in self.epoch_times],
                "final_test_accuracy_percent": round(final_accuracy, 2),
                "all_accuracies_percent": [round(a, 2) for a in self.accuracies],
                "local_steps_completed": self.local_step_counter,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs
            }
            print(f"METRICS_JSON: {json.dumps(metrics)}")
            print("--- End Metrics ---")
    
    def cleanup(self):
        """Clean up resources and notify server"""
        print("Cleaning up...")
        self.training_active = False

        try:
            if self.stub and self.worker_id is not None:
                self.stub.JobFinished(
                    ps_pb2.JobFinishedRequest(worker_id=self.worker_id)
                )
                print(f"Notified server that Worker {self.worker_id} finished")
        except:
            pass
        
        if self.channel:
            self.channel.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Distributed Training Worker')
    parser.add_argument('--server',
                       default=os.environ.get('PARAMETER_SERVER_ADDRESS', 'localhost:8000'),
                       help='Parameter server address')
    parser.add_argument('--worker-name', default='worker-node',
                       help='Worker name for identification')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--sync-steps', type=int, default=1,
                       help='Local steps before syncing with server')

    args = parser.parse_args()
    
    # Create and configure worker
    worker = DistributedWorker(
        server_address=args.server,
        worker_name=args.worker_name
    )
    
    worker.batch_size = args.batch_size
    worker.learning_rate = args.lr
    worker.num_epochs = args.epochs
    worker.local_steps_per_sync = args.sync_steps
    
    try:
        # Setup phases
        worker.connect_to_server()
        worker.setup_model()
        worker.setup_data()
        
        print(f"Worker setup complete. Starting training with:")
        print(f"  - Batch size: {worker.batch_size}")
        print(f"  - Learning rate: {worker.learning_rate}")
        print(f"  - Sync frequency: every {worker.local_steps_per_sync} steps")
        
        # Run training
        worker.run_training()
        
    except Exception as e:
        print(f"Worker failed: {e}")
        return 1
    
    print("Worker completed successfully!")
    return 0

if __name__ == '__main__':
    exit(main())