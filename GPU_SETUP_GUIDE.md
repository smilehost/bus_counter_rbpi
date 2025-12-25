# GPU Setup Guide for Windows

## Current Issue
Your system is currently running PyTorch CPU-only version (`2.9.1+cpu`), which is why the project defaults to CPU instead of GPU.

## Solution: Install PyTorch with CUDA Support

### Step 1: Check Your CUDA Version
First, check what CUDA version you have installed:
```bash
nvidia-smi
```
Look for the "CUDA Version" in the top right corner.

### Step 2: Uninstall Current PyTorch
```bash
pip uninstall torch torchvision
```

### Step 3: Install PyTorch with CUDA Support

#### For CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### If you don't have CUDA installed:
1. Download and install CUDA Toolkit from NVIDIA: https://developer.nvidia.com/cuda-downloads
2. Then install PyTorch with CUDA using one of the commands above

### Step 4: Verify Installation
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

You should see something like:
```
PyTorch version: 2.9.1+cu118
CUDA available: True
```

### Step 5: Test the Project
Now run the project again:
```bash
python example_usage.py
```

The project should now automatically detect and use your GPU.

## Platform-Specific Behavior

After proper GPU installation, the project will automatically:

- **On Windows with CUDA GPU**: Use GPU acceleration with `.pt` models
- **On Raspberry Pi 5**: Use CPU with `.pt` models
- **On other platforms**: Use CPU with `.pt` models (or GPU if CUDA available)

## Troubleshooting

### If CUDA is still not available after installation:
1. Check NVIDIA driver version: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Make sure you're using the correct PyTorch wheel for your CUDA version

### If you get DLL errors:
1. Make sure NVIDIA drivers are up to date
2. Reinstall CUDA Toolkit
3. Reinstall PyTorch with correct CUDA version

### If performance is still slow:
1. Verify GPU is actually being used (check device info output)
2. Monitor GPU usage with Task Manager or `nvidia-smi`
3. Check that your model is moved to GPU: `model.to('cuda')`