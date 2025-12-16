#!/bin/bash
# Script to regenerate gRPC proto files

set -e

echo "Regenerating proto files..."

# Navigate to project root
cd "$(dirname "$0")"

# Generate Python gRPC stubs from proto file
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    src/communication/ps.proto

echo "Proto files regenerated successfully!"
echo "Generated files:"
echo "  - src/communication/ps_pb2.py"
echo "  - src/communication/ps_pb2_grpc.py"
