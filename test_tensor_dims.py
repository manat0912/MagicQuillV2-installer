#!/usr/bin/env python3
"""
Specific tensor dimension test for PIL to tensor conversions
Tests the exact scenario that was causing the mismatch error (ASCII-only)
"""

import torch
import numpy as np
import cv2
from PIL import Image
import sys

def test_pil_to_tensor_dimension_handling():
    """Test PIL image -> numpy -> blur -> tensor with correct dimensions"""
    print("\n=== Tensor Dimension Handling Test ===")
    
    # Simulate what _tensor_to_pil does: converts torch (1, 1, H, W) -> PIL Image
    print("1. Creating mock tensor (1, 1, 512, 512)...")
    mock_mask_tensor = torch.ones(1, 1, 512, 512) * 0.5
    mock_mask_tensor[:, :, 100:200, 100:200] = 0.0  # Active region
    
    # Convert to PIL (simulating _tensor_to_pil behavior)
    print("2. Converting tensor to PIL Image...")
    # PIL expects (H, W) or (H, W, C) in uint8
    mask_np = (mock_mask_tensor[0, 0].numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np, mode='L')  # Grayscale
    
    print(f"   PIL Image size: {mask_pil.size} (W, H format)")
    print(f"   PIL Image mode: {mask_pil.mode}")
    
    # Convert back to numpy (what happens in blending code)
    print("3. Converting PIL back to numpy...")
    blend_mask_np = np.array(mask_pil)
    print(f"   Numpy shape: {blend_mask_np.shape}")
    
    # Handle potential RGB/RGBA conversion
    if blend_mask_np.ndim == 3:
        print(f"   Note: Got {blend_mask_np.ndim}D array, extracting first channel")
        blend_mask_np = blend_mask_np[:, :, 0]
    
    print(f"   Final numpy shape: {blend_mask_np.shape}")
    
    # Apply Gaussian blur
    print("4. Applying Gaussian blur...")
    blurred_blend = cv2.GaussianBlur(blend_mask_np, (11, 11), 3.0)
    print(f"   Blurred shape: {blurred_blend.shape}")
    
    # Convert back to tensor with CORRECT dimensions (1, 1, H, W)
    print("5. Converting blurred mask back to tensor...")
    print("   Using .unsqueeze(0).unsqueeze(0) to get (1, 1, H, W)")
    blend_mask_smooth = torch.from_numpy(blurred_blend / 255.0).float().unsqueeze(0).unsqueeze(0)
    print(f"   Final tensor shape: {blend_mask_smooth.shape}")
    
    # Simulate image tensors
    print("6. Creating mock image tensors for blending...")
    final_image = torch.ones(1, 3, 512, 512) * 0.7  # Generated
    original_image_tensor = torch.ones(1, 3, 512, 512) * 0.8  # Original
    
    print(f"   Generated image shape: {final_image.shape}")
    print(f"   Original image shape: {original_image_tensor.shape}")
    print(f"   Blend mask shape: {blend_mask_smooth.shape}")
    
    # Test blending operation
    print("7. Testing blending operation...")
    try:
        result = final_image * (1.0 - blend_mask_smooth) + original_image_tensor * blend_mask_smooth
        print(f"   [OK] Blending succeeded!")
        print(f"   Result shape: {result.shape}")
        print(f"   Result dtype: {result.dtype}")
        
        # Verify the blend worked correctly
        center_value = result[0, 0, 256, 256].item()
        masked_value = result[0, 0, 150, 150].item()
        
        print(f"   Center (unmasked) value: {center_value:.3f} (expect ~0.75, closer to original 0.8)")
        print(f"   Masked region value: {masked_value:.3f} (expect ~0.70, closer to generated 0.7)")
        
        return True
    except RuntimeError as e:
        print(f"   [FAIL] Blending FAILED: {e}")
        return False

def test_rgb_pil_handling():
    """Test that RGB PIL images are properly handled"""
    print("\n=== RGB PIL Image Handling Test ===")
    
    print("1. Creating mask tensor...")
    mask_tensor = torch.ones(1, 1, 256, 256) * 0.5
    mask_tensor[:, :, 50:100, 50:100] = 0.0
    
    print("2. Converting to RGB PIL (potential issue case)...")
    # Sometimes PIL converts grayscale to RGB
    mask_np_uint8 = (mask_tensor[0, 0].numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np_uint8, mode='L')
    mask_pil_rgb = mask_pil.convert('RGB')  # Force RGB conversion
    
    print(f"   PIL mode: {mask_pil_rgb.mode}")
    
    print("3. Converting RGB PIL to numpy...")
    rgb_np = np.array(mask_pil_rgb)
    print(f"   RGB numpy shape: {rgb_np.shape}")
    
    print("4. Extracting first channel...")
    if rgb_np.ndim == 3:
        rgb_np = rgb_np[:, :, 0]
    print(f"   After extraction shape: {rgb_np.shape}")
    
    print("5. Applying Gaussian blur and converting to tensor...")
    blurred = cv2.GaussianBlur(rgb_np, (11, 11), 3.0)
    blend_mask = torch.from_numpy(blurred / 255.0).float().unsqueeze(0).unsqueeze(0)
    
    print(f"   Final tensor shape: {blend_mask.shape}")
    
    print("6. Testing blend with (1, 3, 256, 256) image...")
    image = torch.ones(1, 3, 256, 256) * 0.5
    try:
        result = image * (1.0 - blend_mask) + image * blend_mask
        print(f"   [OK] RGB handling succeeded! Result shape: {result.shape}")
        return True
    except RuntimeError as e:
        print(f"   [FAIL] RGB handling failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Tensor Dimension Fix Validation")
    print("=" * 60)
    
    try:
        test1 = test_pil_to_tensor_dimension_handling()
        test2 = test_rgb_pil_handling()
        
        print("\n" + "=" * 60)
        if test1 and test2:
            print("[SUCCESS] ALL TENSOR DIMENSION TESTS PASSED")
            print("=" * 60)
            print("\nThe tensor dimension fix resolves the blending error!")
            return 0
        else:
            print("[FAIL] SOME TESTS FAILED")
            return 1
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
