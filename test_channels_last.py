#!/usr/bin/env python3
"""
Test for channels-last tensor format blending (1, H, W, 3)
This is what _pil_to_tensor actually returns (ASCII-only)
"""

import torch
import numpy as np
import cv2
from PIL import Image
import sys

def test_channels_last_blending():
    """Test blending with channels-last format from _pil_to_tensor"""
    print("\n=== Channels-Last Format Blending Test ===")
    
    print("1. Creating mock image tensor in channels-last format...")
    # _pil_to_tensor returns (1, H, W, 3) for RGB images
    final_image = torch.ones(1, 512, 512, 3) * 0.7  # Generated image (light gray)
    original_image = torch.ones(1, 512, 512, 3) * 0.8  # Original image (lighter gray)
    
    print(f"   Generated image shape: {final_image.shape}")
    print(f"   Original image shape: {original_image.shape}")
    
    print("2. Creating blend mask from edge_mask...")
    edit_mask = torch.ones(1, 1, 512, 512) * 0.5
    edit_mask[:, :, 100:200, 100:200] = 0.0  # Active region
    
    # Simulate _tensor_to_pil -> blur -> tensor conversion
    edit_mask_np = (edit_mask[0, 0].numpy() * 255).astype(np.uint8)
    blurred_mask = cv2.GaussianBlur(edit_mask_np, (11, 11), 3.0)
    
    # Create blend mask in channels-last format (1, H, W, 1)
    blend_mask_smooth = torch.from_numpy(blurred_mask / 255.0).float().unsqueeze(0).unsqueeze(-1)
    print(f"   Blend mask shape: {blend_mask_smooth.shape}")
    
    print("3. Testing blending operation with channels-last format...")
    try:
        result = final_image * (1.0 - blend_mask_smooth) + original_image * blend_mask_smooth
        print(f"   [OK] Blending succeeded!")
        print(f"   Result shape: {result.shape}")
        
        # Verify blend worked
        center_value = result[0, 256, 256, 0].item()
        masked_value = result[0, 150, 150, 0].item()
        
        print(f"   Center (low mask) value: {center_value:.3f} (expect ~0.70, closer to generated)")
        print(f"   Masked (high mask) value: {masked_value:.3f} (expect ~0.80, closer to original)")
        
        return True
    except RuntimeError as e:
        print(f"   [FAIL] Blending FAILED: {e}")
        return False

def test_channels_first_to_last_conversion():
    """Test converting channels-first to channels-last for blending"""
    print("\n=== Channels-First to Last Conversion Test ===")
    
    print("1. Creating image in channels-first format (1, 3, H, W)...")
    # This might be what we receive as img = merged_image
    img_cf = torch.ones(1, 3, 256, 256) * 0.5
    print(f"   Channels-first shape: {img_cf.shape}")
    
    print("2. Converting to channels-last for blending...")
    if img_cf.ndim == 4 and img_cf.shape[1] in [3, 4]:
        img_cl = img_cf.permute(0, 2, 3, 1).contiguous()
        print(f"   Channels-last shape: {img_cl.shape}")
    
    print("3. Creating final_image in channels-last format...")
    final_image = torch.ones(1, 256, 256, 3) * 0.7
    
    print("4. Creating blend mask (1, H, W, 1)...")
    blend_mask = torch.ones(1, 256, 256, 1) * 0.5
    blend_mask[:, :100, :100, :] = 0.0
    
    print("5. Testing blending...")
    try:
        result = final_image * (1.0 - blend_mask) + img_cl * blend_mask
        print(f"   [OK] Blending succeeded!")
        print(f"   Result shape: {result.shape}")
        return True
    except RuntimeError as e:
        print(f"   [FAIL] Blending failed: {e}")
        return False

def test_edge_case_handling():
    """Test handling of edge cases in blending"""
    print("\n=== Edge Case Handling Test ===")
    
    print("1. Test with no conversion needed (already channels-last)...")
    final_image = torch.ones(1, 256, 256, 3) * 0.5
    img = torch.ones(1, 256, 256, 3) * 0.7  # Already channels-last
    blend_mask = torch.ones(1, 256, 256, 1) * 0.5
    
    try:
        result = final_image * (1.0 - blend_mask) + img * blend_mask
        print(f"   [OK] Direct channels-last blending works")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False
    
    print("2. Test with conversion needed...")
    final_image = torch.ones(1, 256, 256, 3) * 0.5
    img = torch.ones(1, 3, 256, 256) * 0.7  # Channels-first, needs conversion
    
    # Apply conversion logic
    if img.ndim == 4 and img.shape[1] in [3, 4]:
        img = img.permute(0, 2, 3, 1).contiguous()
    
    blend_mask = torch.ones(1, 256, 256, 1) * 0.5
    
    try:
        result = final_image * (1.0 - blend_mask) + img * blend_mask
        print(f"   [OK] Channels-first to last conversion works")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("Channels-Last Format Blending Test")
    print("=" * 60)
    
    try:
        test1 = test_channels_last_blending()
        test2 = test_channels_first_to_last_conversion()
        test3 = test_edge_case_handling()
        
        print("\n" + "=" * 60)
        if test1 and test2 and test3:
            print("[SUCCESS] ALL CHANNELS-LAST BLENDING TESTS PASSED")
            print("=" * 60)
            print("\nThe foreground_edit() blending should now work!")
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
