#!/bin/bash

# WanVideo Model Download Script for ComfyUI
# This script downloads models, LoRAs, and installs custom nodes

set -e  # Exit on error

# Start timing
START_TIME=$(date +%s)

# Define base ComfyUI path - hardcoded for consistency
COMFYUI_PATH="/workspace/runpod-slim/ComfyUI"
MODELS_PATH="$COMFYUI_PATH/models"
CUSTOM_NODES_PATH="$COMFYUI_PATH/custom_nodes"

echo ""
echo "========================================"
echo "ComfyUI Setup Script"
echo "========================================"
echo "ComfyUI Path: $COMFYUI_PATH"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p "$MODELS_PATH/vae"
mkdir -p "$MODELS_PATH/unet"
mkdir -p "$MODELS_PATH/loras/lightx"
mkdir -p "$MODELS_PATH/loras/WanVideo"
mkdir -p "$MODELS_PATH/loras"
mkdir -p "$MODELS_PATH/text_encoders"
mkdir -p "$MODELS_PATH/clip_vision"
mkdir -p "$MODELS_PATH/model_patches"
mkdir -p "$MODELS_PATH/ultralytics/bbox"
mkdir -p "$MODELS_PATH/sams"
mkdir -p "$CUSTOM_NODES_PATH"

# Function to download file
download_file() {
    local url=$1
    local output_path=$2
    local filename=$(basename "$url" | cut -d'?' -f1)
    
    # Check if file already exists
    if [ -f "$output_path/$filename" ]; then
        echo "  ✓ $filename already exists, skipping..."
        return 0
    fi
    
    echo "Downloading: $filename"
    echo "  to: $output_path"
    
    wget -q --show-progress \
        -O "$output_path/$filename" \
        "$url"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully downloaded"
    else
        echo "  ✗ Failed to download"
        exit 1
    fi
}

# Function to clone custom node
clone_custom_node() {
    local url=$1
    local node_name=$(basename "$url" .git)
    local target_path="$CUSTOM_NODES_PATH/$node_name"
    
    if [ -d "$target_path" ]; then
        echo "  ✓ $node_name already exists, skipping..."
        return 0
    fi
    
    echo "Cloning: $node_name"
    git clone "$url" "$target_path"
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully cloned"
    else
        echo "  ✗ Failed to clone"
        exit 1
    fi
    
    # Install requirements if they exist
    if [ -f "$target_path/requirements.txt" ]; then
        echo "  Installing dependencies for $node_name..."
        pip install -r "$target_path/requirements.txt" --break-system-packages 2>&1 | grep -E "^(Successfully|Collecting|ERROR)" || true
        if [ $? -eq 0 ]; then
            echo "  ✓ Dependencies installed"
        else
            echo "  ⚠ Warning: Some dependencies may have failed, continuing..."
        fi
    else
        echo "  ℹ No requirements.txt found, skipping dependencies"
    fi
}

echo ""
echo "========================================"
echo "Downloading Models (Parallel - 5 concurrent)..."
echo "========================================"
echo ""

# Array of models to download: (url|output_path)
declare -a MODELS=(
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors|$MODELS_PATH/vae"
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors|$MODELS_PATH/vae"
    "https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/resolve/main/Wan2.2-Animate-14B-Q8_0.gguf|$MODELS_PATH/unet"
    "https://huggingface.co/wangkanai/wan21-lightx2v-i2v-14b-480p/resolve/main/loras/wan/wan21-lightx2v-i2v-14b-480p-cfg-step-distill-rank256-bf16.safetensors|$MODELS_PATH/loras/lightx"
    "https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/OtherLoRa's/Wan14B_RealismBoost.safetensors|$MODELS_PATH/loras/WanVideo"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors|$MODELS_PATH/loras"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors|$MODELS_PATH/text_encoders"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors|$MODELS_PATH/clip_vision"
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors|$MODELS_PATH/text_encoders"
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors|$MODELS_PATH/unet"
    "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors|$MODELS_PATH/model_patches"
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt|$MODELS_PATH/ultralytics/bbox"
    "https://huggingface.co/GritTin/LoraStableDiffusion/resolve/c7766cc3c9b8b4f914932ce27f1cd48f25434636/Eyeful_v2-Paired.pt|$MODELS_PATH/ultralytics/bbox"
    "https://huggingface.co/guon/hand-eyes/resolve/ef8ba0842cdc5ba8cf8dacee20dc95e0330c405f/lips_v1.pt|$MODELS_PATH/ultralytics/bbox"
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth|$MODELS_PATH/sams"
)

# Function to download model with parallel support
download_model_parallel() {
    local url=$1
    local output_path=$2
    local filename=$(basename "$url" | cut -d'?' -f1)
    
    # Check if file already exists
    if [ -f "$output_path/$filename" ]; then
        echo "✓ $filename"
        return 0
    fi
    
    echo "⏳ Downloading $filename..."
    wget -q --show-progress \
        -O "$output_path/$filename" \
        "$url" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ $filename"
    else
        echo "✗ $filename (failed)"
        return 1
    fi
}

export -f download_model_parallel

# Download all models in parallel (5 at a time)
for model_pair in "${MODELS[@]}"; do
    IFS='|' read -r url path <<< "$model_pair"
    download_model_parallel "$url" "$path" &
    
    # Limit to 5 parallel downloads
    if [ $(jobs -r -p | wc -l) -ge 5 ]; then
        wait -n
    fi
done

# Wait for all remaining downloads to complete
wait
echo ""
echo "✓ All model downloads completed"

echo ""
echo "========================================"
echo "Setting up Ollama..."
echo "========================================"
echo ""

# Check if ollama is already running
if pgrep -x "ollama" > /dev/null; then
    echo "✓ Ollama is already running"
else
    # Check if ollama is installed
    if command -v ollama &> /dev/null; then
        echo "✓ Ollama is installed but not running, starting service..."
        ollama serve &
        sleep 3
        echo "✓ Ollama service started"
    else
        echo "Ollama not found, installing..."
        echo "Installing zstd dependency..."
        apt-get update -qq && apt-get install -y zstd > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ zstd installed"
        else
            echo "✗ Failed to install zstd"
            exit 1
        fi
        
        echo "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        if [ $? -eq 0 ]; then
            echo "✓ Ollama installed successfully"
            echo "Starting Ollama service..."
            ollama serve &
            sleep 3
            echo "✓ Ollama service started"
        else
            echo "✗ Failed to install Ollama"
            exit 1
        fi
    fi
fi

echo ""
echo "========================================"
echo "Installing Custom Nodes..."
echo "========================================"
echo ""

# Array of custom node repositories
declare -a CUSTOM_NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/city96/ComfyUI-GGUF"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
    "https://github.com/kael558/ComfyUI-GGUF-FantasyTalking"
    "https://github.com/ClownsharkBatwing/RES4LYF"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler"
    "https://github.com/chrisgoringe/cg-use-everywhere"
    "https://github.com/ltdrdata/ComfyUI-Impact-Subpack"
    "https://github.com/aadityamundhalia/ComfyUI-ollama-aditya"
)

# Clone each custom node
for node_url in "${CUSTOM_NODES[@]}"; do
    echo "=== Installing $(basename "$node_url") ==="
    clone_custom_node "$node_url"
    echo ""
done

echo ""
echo "========================================"
echo "Pulling Ollama Models..."
echo "========================================"
echo ""

# Start ollama service if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 3
fi

# Pull the qwen3-vl model
echo "Pulling qwen3-vl:8b model..."
ollama pull qwen3-vl:8b
if [ $? -eq 0 ]; then
    echo "✓ qwen3-vl:8b model installed successfully"
else
    echo "✗ Failed to pull qwen3-vl:8b model"
    exit 1
fi

echo ""
echo "========================================"
echo "Downloading Workflows..."
echo "========================================"
echo ""

# Create workflows directory if it doesn't exist
WORKFLOWS_PATH="$COMFYUI_PATH/user/default/workflows"
mkdir -p "$WORKFLOWS_PATH"

# Download workflows
echo "Downloading z-image controlnet workflow..."
wget --show-progress -O "$WORKFLOWS_PATH/z-image-controlnet.json" \
    "https://raw.githubusercontent.com/aadityamundhalia/comfyui_workflows/refs/heads/main/z-image%20controlnet.json"
if [ $? -eq 0 ]; then
    echo "✓ z-image controlnet workflow downloaded"
else
    echo "⚠ Note: z-image controlnet workflow download failed (may not exist or network issue)"
fi

echo "Downloading wan_animate workflow..."
wget --show-progress -O "$WORKFLOWS_PATH/wan_animate.json" \
    "https://raw.githubusercontent.com/aadityamundhalia/comfyui_workflows/refs/heads/main/wan_animate.json"
if [ $? -eq 0 ]; then
    echo "✓ wan_animate workflow downloaded"
else
    echo "⚠ Note: wan_animate workflow download failed (may not exist or network issue)"
fi

# Check what was actually downloaded
echo ""
echo "Workflows in $WORKFLOWS_PATH:"
ls -lh "$WORKFLOWS_PATH" 2>/dev/null | tail -n +2 || echo "  (No workflows found)"

echo ""
echo "========================================"
echo "⚠️  ComfyUI Restart Required"
echo "========================================"
echo ""
echo "Please restart ComfyUI to load all the new:"
echo "  • Custom nodes"
echo "  • Models"
echo "  • Workflows"
echo ""
echo "If running locally: Press Ctrl+C and restart"
echo "If running on RunPod: Restart the pod from the dashboard"
echo ""

echo ""
echo "========================================"
echo "✓ All downloads and installations completed successfully!"
echo "========================================"
echo ""
echo "Downloaded models:"
echo "  • VAE: $MODELS_PATH/vae"
echo "  • UNet: $MODELS_PATH/unet"
echo "  • LoRAs: $MODELS_PATH/loras"
echo "  • Text Encoder: $MODELS_PATH/text_encoders"
echo "  • CLIP Vision: $MODELS_PATH/clip_vision"
echo "  • Model Patches: $MODELS_PATH/model_patches"
echo "  • YOLOv8 Detection: $MODELS_PATH/ultralytics/bbox"
echo "  • SAM Models: $MODELS_PATH/sams"
echo ""
echo "Installed custom nodes:"
echo "  • ComfyUI-Manager"
echo "  • comfyui_controlnet_aux"
echo "  • ComfyUI-GGUF"
echo "  • rgthree-comfy"
echo "  • ComfyUI-VideoHelperSuite"
echo "  • ComfyUI-Frame-Interpolation"
echo "  • ComfyUI-GGUF-FantasyTalking"
echo "  • RES4LYF"
echo "  • ComfyUI-Impact-Pack"
echo "  • ComfyUI-Easy-Use"
echo "  • ComfyUI-SeedVR2_VideoUpscaler"
echo "  • cg-use-everywhere"
echo "  • ComfyUI-Impact-Subpack"
echo "  • ComfyUI-ollama-aditya"
echo ""
echo "Ollama models:"
echo "  • qwen3-vl:8b"
echo ""
echo "Workflows:"
echo "  • z-image-controlnet.json"
echo "  • wan_animate.json"
echo ""

# Calculate execution time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "========================================"
echo "⏱️  Total Execution Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================"
