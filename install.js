module.exports = {
  run: [

    // Clone MagicQuillV2 repository
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/manat0912/MagicQuillV2.git app",
        ]
      }
    },
    // Install Python requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          "HF_HOME": "{{path.resolve(cwd, 'cache/HF_HOME')}}",
          "TORCH_HOME": "{{path.resolve(cwd, 'cache/TORCH_HOME')}}",
          "GRADIO_TEMP_DIR": "{{path.resolve(cwd, 'cache/GRADIO_TEMP_DIR')}}"
        },
        path: "app",
        message: [
          "uv pip install gradio devicetorch",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Run PyTorch/CUDA setup (without modifying torch.js)
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    // Download MagicQuill V2 models using huggingface_hub in venv (no token)
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          "HF_HOME": "{{path.resolve(cwd, 'cache/HF_HOME')}}",
          "TORCH_HOME": "{{path.resolve(cwd, 'cache/TORCH_HOME')}}",
          "GRADIO_TEMP_DIR": "{{path.resolve(cwd, 'cache/GRADIO_TEMP_DIR')}}"
        },
        path: "app",
        message: [
          "python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='LiuZichen/MagicQuillV2-models', local_dir='models', token=False)\""
        ]
      }
    },
    // Download Q5_K_M GGUF Kontext model
    {
      method: "fs.download",
      params: {
        url: "https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/resolve/main/flux1-kontext-dev-Q5_K_M.gguf",
        path: "app/models/v2_ckpt/split_files/diffusion_models/flux1-kontext-dev-Q5_K_M.gguf"
      }
    },
    // Download CLIP-L text encoder
    {
      method: "fs.download",
      params: {
        url: "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        path: "app/models/v2_ckpt/split_files/text_encoders/clip_l.safetensors"
      }
    },
    // Download T5XXL FP8 text encoder
    {
      method: "fs.download",
      params: {
        url: "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors",
        path: "app/models/v2_ckpt/split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"
      }
    },
    // Download Lumina ae VAE
    {
      method: "fs.download",
      params: {
        url: "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors",
        path: "app/models/v2_ckpt/split_files/vae/ae.safetensors"
      }
    },
    // Download precompiled llama.cpp binaries
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          "HF_HOME": "{{path.resolve(cwd, 'cache/HF_HOME')}}",
          "TORCH_HOME": "{{path.resolve(cwd, 'cache/TORCH_HOME')}}",
          "GRADIO_TEMP_DIR": "{{path.resolve(cwd, 'cache/GRADIO_TEMP_DIR')}}"
        },
        path: "app",
        message: [
          "python ../download_llama.py"
        ]
      }
    },
    // Download quantized model and rename it (no token)
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          "HF_HOME": "{{path.resolve(cwd, 'cache/HF_HOME')}}",
          "TORCH_HOME": "{{path.resolve(cwd, 'cache/TORCH_HOME')}}",
          "GRADIO_TEMP_DIR": "{{path.resolve(cwd, 'cache/GRADIO_TEMP_DIR')}}"
        },
        path: "app",
        message: [
          "python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='PsiPi/liuhaotian_llava-v1.5-13b-GGUF', filename='llava-v1.5-13b-Q5_K_M.gguf', local_dir='../llama.cpp/models', token=False)\"",
          "python -c \"import os; src='../llama.cpp/models/llava-v1.5-13b-Q5_K_M.gguf'; dst='../llama.cpp/models/magicquill-13b-q5km.gguf'; os.path.exists(dst) and os.remove(dst); os.rename(src, dst)\""
        ]
      }
    }
  ]
}
