import urllib.request
import json
import zipfile
import io
import os
import sys

def main():
    # Since download_llama.py is in the workspace root, __file__ points to the root.
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(root_dir, "llama.cpp")
    os.makedirs(os.path.join(dest_dir, "models"), exist_ok=True)

    # Detect CUDA
    use_cuda = False
    try:
        import torch
        if torch.cuda.is_available():
            use_cuda = True
    except Exception:
        pass

    print(f"CUDA status: {'ENABLED' if use_cuda else 'DISABLED'}")

    bin_url = None
    dll_url = None
    fallback_tag = "b9616"

    # Try fetching the latest release from GitHub API
    try:
        print("Fetching latest llama.cpp release from GitHub API...")
        req = urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode())
            tag_name = data.get("tag_name", fallback_tag)
            print(f"Latest release found: {tag_name}")
            
            suffix = "win-cuda-12.4-x64.zip" if use_cuda else "win-llvm-x64.zip"
            for a in data.get("assets", []):
                if suffix in a["name"]:
                    if "cudart" in a["name"]:
                        dll_url = a["browser_download_url"]
                    else:
                        bin_url = a["browser_download_url"]
    except Exception as e:
        print(f"GitHub API check failed ({e}). Using fallback build {fallback_tag}.")

    # Fallback URLs if API failed or assets were not found
    if not bin_url:
        if use_cuda:
            bin_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-win-cuda-12.4-x64.zip"
            dll_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/cudart-llama-bin-win-cuda-12.4-x64.zip"
        else:
            bin_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-win-llvm-x64.zip"

    # Download and extract binaries
    try:
        print(f"Downloading binary from: {bin_url}")
        req = urllib.request.Request(bin_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                z.extractall(dest_dir)
        print("Successfully extracted llama.cpp binaries.")

        if dll_url:
            print(f"Downloading CUDA runtime DLL from: {dll_url}")
            req = urllib.request.Request(dll_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r:
                with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                    z.extractall(dest_dir)
            print("Successfully extracted CUDA runtime DLL.")
            
        print("llama.cpp setup completed successfully!")
    except Exception as e:
        print(f"Error setting up llama.cpp: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
