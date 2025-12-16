import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grpc
import time
import pickle
import threading
from concurrent import futures
import numpy as np
from collections import defaultdict, deque
import copy
import argparse

from communication import ps_pb2
from communication import ps_pb2_grpc

import torch
import torch.nn as nn

# --- Enhanced ResNet-18 for CIFAR-100 ---
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

# --- Server configuration ---
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class ParameterServerServicer(ps_pb2_grpc.ParameterServerServicer):
    """Enhanced Parameter Server with Sync/Async modes and scaling support"""
    
    def __init__(self, mode='sync', total_workers=4, learning_rate=0.1, staleness_bound=5):
        print(f"Initializing Parameter Server in {mode.upper()} mode...")
        
        # --- Server Configuration ---
        self.mode = mode  # 'sync' or 'async'
        self.total_workers = total_workers
        self.learning_rate = learning_rate
        self.staleness_bound = staleness_bound  # For bounded staleness in async
        
        # --- Model Parameters ---
        print("Loading ResNet-18 model for CIFAR-100...")
        model = ResNet18(num_classes=100)
        self.parameters = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        self.param_lock = threading.Lock()
        print(f"Model loaded with {len(self.parameters)} parameter tensors.")
        
        # --- Worker Management ---
        self.registered_workers = {}  # {worker_id: worker_info}
        self.registered_workers_count = 0
        self.registration_lock = threading.Lock()
        self.active_workers = set()
        
        # --- Training State ---
        self.global_step = 0
        self.total_updates = 0
        self.gradients_processed = 0
        
        # --- Synchronous Mode State ---
        self.pending_gradients = {}  # {worker_id: gradients}
        self.gradients_received = 0
        self.sync_lock = threading.Lock()
        
        # --- Asynchronous Mode State ---
        self.gradient_staleness = {}  # {worker_id: staleness_value}
        self.async_updates = 0
        
        # --- Performance Metrics ---
        self.update_times = deque(maxlen=100)  # Track recent update times
        self.start_time = time.time()
        
        print(f"Server initialized for {total_workers} workers in {mode} mode")

    def apply_gradients(self, aggregated_gradients, update_info=""):
        """Apply gradients to model parameters with performance tracking"""
        update_start = time.time()
        
        with self.param_lock:
            for param_name, gradient in aggregated_gradients.items():
                if param_name in self.parameters:
                    self.parameters[param_name] = self.parameters[param_name] - self.learning_rate * gradient
                else:
                    print(f"Warning: Parameter {param_name} not found in model")
            
            self.global_step += 1
            self.total_updates += 1
            
        update_time = time.time() - update_start
        self.update_times.append(update_time)
        
        print(f"Applied gradients (Step {self.global_step}): {update_info}, Update time: {update_time:.4f}s")

    def aggregate_gradients_sync(self, worker_gradients):
        """Synchronous gradient aggregation using averaging"""
        if not worker_gradients:
            return {}
        
        param_names = list(worker_gradients[0].keys())
        aggregated = {}
        
        for param_name in param_names:
            gradient_sum = None
            valid_workers = 0
            
            for gradients in worker_gradients:
                if param_name in gradients:
                    grad = gradients[param_name]
                    if gradient_sum is None:
                        gradient_sum = grad.copy()
                    else:
                        gradient_sum += grad
                    valid_workers += 1
            
            if valid_workers > 0:
                aggregated[param_name] = gradient_sum / valid_workers
        
        return aggregated

    def apply_gradients_async(self, gradients, worker_id, staleness):
        """Asynchronous gradient application with staleness handling"""
        if staleness > self.staleness_bound:
            print(f"Rejecting gradients from worker {worker_id} (staleness {staleness} > bound {self.staleness_bound})")
            return False
        
        # Apply gradients with staleness compensation
        staleness_weight = max(0.1, 1.0 / (1.0 + staleness * 0.1))  # Reduce weight for stale gradients
        
        weighted_gradients = {}
        for param_name, gradient in gradients.items():
            weighted_gradients[param_name] = gradient * staleness_weight
        
        self.apply_gradients(weighted_gradients, f"Async worker {worker_id}, staleness {staleness}")
        self.async_updates += 1
        return True

    # --- gRPC Service Methods ---
    
    def RegisterWorker(self, request, context):
        """Register worker with enhanced tracking"""
        with self.registration_lock:
            worker_id = self.registered_workers_count
            self.registered_workers_count += 1
            
            worker_info = {
                'worker_id': worker_id,
                'worker_ip': request.worker_ip,
                'registered_at': time.time(),
                'last_seen': time.time()
            }
            self.registered_workers[worker_id] = worker_info
            self.active_workers.add(worker_id)
        
        print(f"[RegisterWorker] Worker {worker_id} from {request.worker_ip} registered")
        print(f"Active workers: {len(self.active_workers)}/{self.total_workers}")
        
        return ps_pb2.RegisterReply(
            worker_id=worker_id,
            total_workers=self.total_workers
        )

    def FetchParameters(self, request, context):
        """Fetch parameters with worker activity tracking"""
        worker_id = request.worker_id
        
        # Update worker activity
        if worker_id in self.registered_workers:
            self.registered_workers[worker_id]['last_seen'] = time.time()
        
        with self.param_lock:
            params_bytes = pickle.dumps(self.parameters)
            current_global_step = self.global_step

        print(f"[FetchParameters] Worker {worker_id} fetched parameters (Global step: {current_global_step})")
        
        return ps_pb2.FetchReply(
            parameters=params_bytes,
            global_step=current_global_step
        )
    
    def decompress_gradients(self, compressed_gradients):  # Add this new method
        """Convert float16 gradients back to float32"""
        decompressed = {}
        for name, grad in compressed_gradients.items():
            decompressed[name] = grad.astype(np.float32)
        return decompressed
    
    def PushGradrients(self, request, context):
        """Enhanced gradient handling for both sync and async modes"""
        worker_id = request.worker_id
        local_step = request.local_step
        
        try:
            gradients = pickle.loads(request.gradients)
            gradients = self.decompress_gradients(gradients)
            self.gradients_processed += 1
            
            # Update worker activity
            if worker_id in self.registered_workers:
                self.registered_workers[worker_id]['last_seen'] = time.time()
            
            print(f"[PushGradients] Worker {worker_id}, Local step {local_step}, Mode: {self.mode}")
            
            if self.mode == 'sync':
                return self._handle_sync_gradients(worker_id, gradients, local_step)
            else:  # async mode
                return self._handle_async_gradients(worker_id, gradients, local_step)
                
        except Exception as e:
            print(f"Error processing gradients from worker {worker_id}: {e}")
            return ps_pb2.PushReply(received=False)

    def _handle_sync_gradients(self, worker_id, gradients, local_step):
        """Handle synchronous gradient updates"""
        with self.sync_lock:
            self.pending_gradients[worker_id] = gradients
            self.gradients_received += 1
            
            print(f"  > Sync: {self.gradients_received}/{self.total_workers} workers ready")
            
            if self.gradients_received >= self.total_workers:
                # All workers ready - aggregate and apply
                worker_gradients = list(self.pending_gradients.values())
                aggregated_gradients = self.aggregate_gradients_sync(worker_gradients)
                
                if aggregated_gradients:
                    self.apply_gradients(aggregated_gradients, f"Sync update from {len(worker_gradients)} workers")
                
                # Reset for next round
                self.pending_gradients.clear()
                self.gradients_received = 0
                
                print(f"  > Synchronous update completed! Global step: {self.global_step}")
            else:
                print(f"  > Waiting for {self.total_workers - self.gradients_received} more workers...")
        
        return ps_pb2.PushReply(received=True)

    def _handle_async_gradients(self, worker_id, gradients, local_step):
        """Handle asynchronous gradient updates"""
        # Calculate staleness
        current_global_step = self.global_step
        staleness = current_global_step - local_step
        
        # Apply gradients immediately if not too stale
        success = self.apply_gradients_async(gradients, worker_id, staleness)
        
        # Track staleness statistics
        self.gradient_staleness[worker_id] = staleness
        
        print(f"  > Async: Applied gradients with staleness {staleness}")
        
        return ps_pb2.PushReply(received=success)

    def JobFinished(self, request, context):
        """Handle job completion with comprehensive statistics"""
        worker_id = request.worker_id
        print(f"[JobFinished] Worker {worker_id} completed")
        
        # Remove from active workers
        self.active_workers.discard(worker_id)
        
        # Print statistics when all workers finish
        if len(self.active_workers) == 0:
            self._print_final_statistics()
        
        return ps_pb2.JobFinishedReply(message="Acknowledged")

    def _print_final_statistics(self):
        """Print comprehensive training statistics"""
        total_time = time.time() - self.start_time
        avg_update_time = np.mean(self.update_times) if self.update_times else 0

        print(f"\n{'='*50}")
        print(f"PARAMETER SERVER FINAL STATISTICS")
        print(f"{'='*50}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Total workers: {self.total_workers}")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Global steps completed: {self.global_step}")
        print(f"Total parameter updates: {self.total_updates}")
        print(f"Gradients processed: {self.gradients_processed}")
        print(f"Average update time: {avg_update_time:.4f} seconds")

        if self.mode == 'async':
            if self.gradient_staleness:
                avg_staleness = np.mean(list(self.gradient_staleness.values()))
                max_staleness = max(self.gradient_staleness.values())
                print(f"Async updates: {self.async_updates}")
                print(f"Average gradient staleness: {avg_staleness:.2f}")
                print(f"Maximum staleness observed: {max_staleness}")

        print(f"Updates per second: {self.total_updates / total_time:.2f}")
        print(f"{'='*50}\n")

        # Output structured JSON metrics for easy parsing
        import json
        metrics = {
            "type": "SERVER_FINAL_METRICS",
            "mode": self.mode,
            "total_workers": self.total_workers,
            "total_training_time_seconds": round(total_time, 2),
            "global_steps_completed": self.global_step,
            "total_parameter_updates": self.total_updates,
            "gradients_processed": self.gradients_processed,
            "average_update_time_seconds": round(avg_update_time, 4),
            "updates_per_second": round(self.total_updates / total_time, 2),
            "learning_rate": self.learning_rate
        }

        if self.mode == 'async' and self.gradient_staleness:
            metrics["async_updates"] = self.async_updates
            metrics["average_gradient_staleness"] = round(np.mean(list(self.gradient_staleness.values())), 2)
            metrics["max_staleness_observed"] = max(self.gradient_staleness.values())

        print(f"METRICS_JSON: {json.dumps(metrics)}")
        print("--- End Server Metrics ---")

def serve(mode='sync', total_workers=4, learning_rate=0.1, port=8000):
    """Start the parameter server with specified configuration"""
    options = [
        ('grpc.max_receive_message_length', 500 * 1024 * 1024),  # 500 MB
        ('grpc.max_send_message_length', 500 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True)
    ]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=20),  # Increase for more workers
        options=options
    )
    
    servicer = ParameterServerServicer(
        mode=mode, 
        total_workers=total_workers,
        learning_rate=learning_rate
    )
    
    ps_pb2_grpc.add_ParameterServerServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"--- Parameter Server listening on port {port} ---")
    print(f"Mode: {mode.upper()}, Workers: {total_workers}, LR: {learning_rate}")
    
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        print("Server shutting down...")
        server.stop(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced Parameter Server')
    parser.add_argument('--mode', choices=['sync', 'async'],
                       default=os.environ.get('SERVER_MODE', 'sync'),
                       help='Training mode: synchronous or asynchronous')
    parser.add_argument('--workers', type=int,
                       default=int(os.environ.get('TOTAL_WORKERS_EXPECTED', '4')),
                       help='Expected number of workers (1-32)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--port', type=int,
                       default=int(os.environ.get('SERVER_PORT', '8000')),
                       help='Server port')
    parser.add_argument('--staleness-bound', type=int, default=5,
                       help='Maximum staleness for async mode')

    args = parser.parse_args()

    # Validate worker count
    if not (1 <= args.workers <= 32):
        print("Error: Number of workers must be between 1 and 32")
        exit(1)

    serve(
        mode=args.mode,
        total_workers=args.workers,
        learning_rate=args.lr,
        port=args.port
    )