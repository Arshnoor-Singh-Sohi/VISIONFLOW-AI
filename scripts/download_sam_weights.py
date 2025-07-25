#!/usr/bin/env python3
"""
VisionFlow AI - SAM Model Weights Download Script
================================================

This script downloads the pre-trained weights for Meta's Segment Anything Model (SAM).
Think of this as downloading the "brain" of your image segmentation system - these
weights contain all the learned knowledge that allows SAM to identify objects in images.

SAM comes in three sizes:
- ViT-H (Huge): Most accurate but largest (2.6GB) - best for production
- ViT-L (Large): Good balance (1.2GB) - good for most use cases  
- ViT-B (Base): Fastest but least accurate (375MB) - good for development

This script intelligently downloads the appropriate model based on your system
capabilities and intended use case.
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from typing import Dict, Tuple
import argparse
from tqdm import tqdm

# Add backend to path for configuration access
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    from backend.config import get_settings
    from backend.utils.helpers import format_bytes
except ImportError:
    print("Warning: Could not import VisionFlow configuration. Using defaults.")
    get_settings = None
    format_bytes = lambda x: f"{x / (1024**3):.1f} GB"

# SAM model information - this is like a catalog of available models
SAM_MODELS = {
    "vit_h": {
        "name": "SAM ViT-H (Huge)",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size_mb": 2656,
        "sha256": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
        "description": "Highest accuracy, best for production use",
        "min_ram_gb": 8,
        "inference_time": "slow"
    },
    "vit_l": {
        "name": "SAM ViT-L (Large)", 
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size_mb": 1249,
        "sha256": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
        "description": "Good balance of speed and accuracy",
        "min_ram_gb": 4,
        "inference_time": "medium"
    },
    "vit_b": {
        "name": "SAM ViT-B (Base)",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", 
        "filename": "sam_vit_b_01ec64.pth",
        "size_mb": 375,
        "sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
        "description": "Fastest inference, good for development",
        "min_ram_gb": 2,
        "inference_time": "fast"
    }
}


def get_system_info() -> Dict[str, any]:
    """
    Analyze system capabilities to recommend the best SAM model.
    
    This function acts like a hardware inspector, checking what your
    computer can handle so we can recommend the right model size.
    """
    try:
        import psutil
        
        # Get memory information
        memory = psutil.virtual_memory()
        available_ram_gb = memory.available / (1024**3)
        total_ram_gb = memory.total / (1024**3)
        
        # Get disk space
        disk = psutil.disk_usage('.')
        available_disk_gb = disk.free / (1024**3)
        
        # Check if we have GPU
        gpu_available = False
        gpu_memory_gb = 0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_available = True
                gpu_memory_gb = max(gpu.memoryTotal for gpu in gpus) / 1024
        except ImportError:
            # Try nvidia-smi as fallback
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_memory_gb = int(result.stdout.strip()) / 1024
                    gpu_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        return {
            "total_ram_gb": total_ram_gb,
            "available_ram_gb": available_ram_gb,
            "available_disk_gb": available_disk_gb,
            "gpu_available": gpu_available,
            "gpu_memory_gb": gpu_memory_gb
        }
        
    except ImportError:
        print("Warning: psutil not available. Cannot analyze system capabilities.")
        return {
            "total_ram_gb": 8.0,  # Reasonable defaults
            "available_ram_gb": 4.0,
            "available_disk_gb": 10.0,
            "gpu_available": False,
            "gpu_memory_gb": 0
        }


def recommend_model(system_info: Dict[str, any], use_case: str = "development") -> str:
    """
    Recommend the best SAM model based on system capabilities and use case.
    
    This is like having a knowledgeable shop assistant who considers your
    needs and budget to recommend the best product for you.
    """
    ram_gb = system_info["available_ram_gb"]
    disk_gb = system_info["available_disk_gb"]
    has_gpu = system_info["gpu_available"]
    
    print(f"System Analysis:")
    print(f"  Available RAM: {ram_gb:.1f} GB")
    print(f"  Available Disk: {disk_gb:.1f} GB") 
    print(f"  GPU Available: {'Yes' if has_gpu else 'No'}")
    print(f"  Use Case: {use_case}")
    print()
    
    # Check if we have enough disk space for any model
    min_disk_needed = min(model["size_mb"] for model in SAM_MODELS.values()) / 1024
    if disk_gb < min_disk_needed:
        raise RuntimeError(f"Insufficient disk space. Need at least {min_disk_needed:.1f} GB")
    
    # Recommend based on use case and system capabilities
    if use_case == "production":
        if ram_gb >= 8 and disk_gb >= 3:
            return "vit_h"  # Best accuracy for production
        elif ram_gb >= 4 and disk_gb >= 2:
            return "vit_l"  # Good compromise
        else:
            return "vit_b"  # Minimal requirements
            
    elif use_case == "development":
        if ram_gb >= 4 and disk_gb >= 2:
            return "vit_l"  # Good for development and testing
        else:
            return "vit_b"  # Fast iteration
            
    else:  # balanced
        if ram_gb >= 6 and disk_gb >= 2:
            return "vit_l"  # Best overall choice
        else:
            return "vit_b"  # Conservative choice


def verify_file_integrity(file_path: str, expected_sha256: str) -> bool:
    """
    Verify that a downloaded file hasn't been corrupted.
    
    This is like checking that a delivered package hasn't been
    damaged in shipping - we compare a "fingerprint" (hash) of
    the file to make sure it's exactly what we expected.
    """
    try:
        sha256_hash = hashlib.sha256()
        
        print(f"Verifying file integrity...")
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        calculated_hash = sha256_hash.hexdigest()
        
        if calculated_hash == expected_sha256:
            print("✓ File integrity verified")
            return True
        else:
            print(f"✗ File integrity check failed!")
            print(f"  Expected: {expected_sha256}")
            print(f"  Got:      {calculated_hash}")
            return False
            
    except Exception as e:
        print(f"✗ File integrity check failed: {e}")
        return False


def download_file(url: str, file_path: str, expected_size_mb: int) -> bool:
    """
    Download a file with progress tracking and error handling.
    
    This function is like a careful delivery service - it downloads
    the file piece by piece, shows you the progress, and handles
    any problems that might occur during the download.
    """
    try:
        print(f"Downloading from: {url}")
        print(f"Saving to: {file_path}")
        
        # Make the request with streaming to handle large files
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size from headers
        file_size = int(response.headers.get('content-length', 0))
        
        if file_size == 0:
            print("Warning: Could not determine file size")
            file_size = expected_size_mb * 1024 * 1024  # Use expected size
        
        # Create progress bar
        progress = tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading"
        )
        
        # Download in chunks
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    progress.update(len(chunk))
        
        progress.close()
        
        # Verify file size
        actual_size = os.path.getsize(file_path)
        expected_size = expected_size_mb * 1024 * 1024
        
        if abs(actual_size - expected_size) > (expected_size * 0.01):  # Allow 1% variance
            print(f"Warning: File size mismatch. Expected ~{format_bytes(expected_size)}, got {format_bytes(actual_size)}")
        
        print("✓ Download completed successfully")
        return True
        
    except requests.RequestException as e:
        print(f"✗ Download failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during download: {e}")
        return False


def download_sam_model(model_key: str, models_dir: str, force: bool = False) -> bool:
    """
    Download and verify a specific SAM model.
    
    This is the main coordinator function that orchestrates the entire
    download process - like a project manager making sure each step
    happens in the right order.
    """
    if model_key not in SAM_MODELS:
        print(f"✗ Unknown model: {model_key}")
        print(f"Available models: {', '.join(SAM_MODELS.keys())}")
        return False
    
    model_info = SAM_MODELS[model_key]
    file_path = os.path.join(models_dir, model_info["filename"])
    
    print(f"📦 {model_info['name']}")
    print(f"   Size: {format_bytes(model_info['size_mb'] * 1024 * 1024)}")
    print(f"   Description: {model_info['description']}")
    print()
    
    # Check if file already exists and is valid
    if os.path.exists(file_path) and not force:
        print(f"Model file already exists: {file_path}")
        
        # Verify integrity of existing file
        if verify_file_integrity(file_path, model_info["sha256"]):
            print("✓ Existing file is valid, skipping download")
            return True
        else:
            print("⚠ Existing file is corrupted, re-downloading...")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Download the file
    if not download_file(model_info["url"], file_path, model_info["size_mb"]):
        return False
    
    # Verify the downloaded file
    if not verify_file_integrity(file_path, model_info["sha256"]):
        print("✗ Downloaded file is corrupted, removing...")
        try:
            os.remove(file_path)
        except OSError:
            pass
        return False
    
    print(f"✓ Successfully downloaded and verified {model_info['name']}")
    return True


def main():
    """
    Main function that coordinates the entire model download process.
    
    This function is like the conductor of an orchestra, making sure
    all the different parts (system analysis, model selection, download)
    work together harmoniously.
    """
    parser = argparse.ArgumentParser(
        description="Download SAM model weights for VisionFlow AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --auto                    # Automatically choose best model
  %(prog)s --model vit_l             # Download specific model
  %(prog)s --all                     # Download all models
  %(prog)s --production              # Optimize for production use
        """
    )
    
    parser.add_argument(
        "--model", 
        choices=list(SAM_MODELS.keys()),
        help="Specific model to download"
    )
    parser.add_argument(
        "--auto", 
        action="store_true",
        help="Automatically choose the best model for your system"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--production", 
        action="store_true",
        help="Optimize model choice for production use"
    )
    parser.add_argument(
        "--models-dir", 
        default="./data/models",
        help="Directory to save models (default: ./data/models)"
    )
    
    args = parser.parse_args()
    
    print("🤖 SAM Model Download Utility")
    print("=" * 40)
    print()
    
    # Get models directory from config if available
    models_dir = args.models_dir
    if get_settings:
        try:
            settings = get_settings()
            models_dir = settings.models_path
        except Exception:
            pass  # Use default
    
    # Determine which models to download
    models_to_download = []
    
    if args.all:
        models_to_download = list(SAM_MODELS.keys())
        print("📋 Downloading all available models...")
        
    elif args.model:
        models_to_download = [args.model]
        print(f"📋 Downloading specific model: {args.model}")
        
    elif args.auto:
        # Analyze system and recommend model
        print("🔍 Analyzing system capabilities...")
        system_info = get_system_info()
        use_case = "production" if args.production else "development"
        recommended = recommend_model(system_info, use_case)
        models_to_download = [recommended]
        
        print(f"📋 Recommended model: {recommended}")
        print(f"   Reason: {SAM_MODELS[recommended]['description']}")
        
    else:
        # Interactive mode - ask user to choose
        print("🤖 Available SAM Models:")
        print()
        
        for key, model in SAM_MODELS.items():
            print(f"  {key}: {model['name']}")
            print(f"      Size: {format_bytes(model['size_mb'] * 1024 * 1024)}")
            print(f"      {model['description']}")
            print(f"      Min RAM: {model['min_ram_gb']} GB")
            print()
        
        # Get system recommendation
        system_info = get_system_info()
        recommended = recommend_model(system_info, "development")
        
        print(f"💡 Recommended for your system: {recommended}")
        print()
        
        choice = input(f"Which model would you like to download? [{recommended}]: ").strip().lower()
        
        if not choice:
            choice = recommended
        
        if choice not in SAM_MODELS:
            print(f"✗ Invalid choice: {choice}")
            return 1
        
        models_to_download = [choice]
    
    print()
    
    # Download selected models
    success_count = 0
    total_count = len(models_to_download)
    
    for i, model_key in enumerate(models_to_download, 1):
        print(f"📥 Downloading model {i}/{total_count}: {model_key}")
        print("-" * 50)
        
        if download_sam_model(model_key, models_dir, args.force):
            success_count += 1
            print()
        else:
            print(f"✗ Failed to download {model_key}")
            print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Download Summary:")
    print(f"   Successful: {success_count}/{total_count}")
    print(f"   Models directory: {models_dir}")
    
    if success_count == total_count:
        print("✅ All downloads completed successfully!")
        
        # Show next steps
        print()
        print("🚀 Next steps:")
        print("   1. Update your .env file with the model path")
        print("   2. Start the SAM service")
        print("   3. Test the integration with a sample image")
        
        return 0
    else:
        print("❌ Some downloads failed. Check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())