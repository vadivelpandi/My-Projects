from PIL import Image

def test_crop_logic():
    w, h = 1000, 1000 # Test with a clean number
    w2, h2 = 1001, 1001 # Test with a prime/odd number to check remainders
    
    check_grid(w, h, "Even 1000x1000")
    check_grid(w2, h2, "Odd 1001x1001")

def check_grid(w, h, label):
    print(f"Testing {label} ({w}x{h})...")
    grid_size = 4
    covered_pixels = 0
    
    # Mock image area
    expected_area = w * h
    
    for r in range(grid_size):
        for c in range(grid_size):
            left = int(c * w / grid_size)
            upper = int(r * h / grid_size)
            right = int((c + 1) * w / grid_size)
            lower = int((r + 1) * h / grid_size)
            
            patch_w = right - left
            patch_h = lower - upper
            
            # Basic sanity
            if patch_w <= 0 or patch_h <= 0:
                print(f"  FAIL: Invalid patch size at {r},{c}: {patch_w}x{patch_h}")
                return
            
            covered_pixels += (patch_w * patch_h)
            # print(f"  Patch {r},{c}: ({left},{upper}) -> ({right},{lower}) = {patch_w}x{patch_h}")

    if covered_pixels == expected_area:
        print(f"  PASS: Full coverage area matches ({covered_pixels}/{expected_area})")
    else:
        print(f"  FAIL: Area mismatch! Got {covered_pixels}, expected {expected_area}")

if __name__ == "__main__":
    test_crop_logic()
