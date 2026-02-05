
from forensic_engine import ForensicEngine
from PIL import Image, ImageDraw
import numpy as np
import json

def create_real_like_image():
    # Random noise (high entropy, high frequency) -> Should be REAL
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def create_ai_like_image():
    # Solid color / Gradient (Low noise, smooth, low freq) -> Should be AI/Suspicious
    img = Image.new('RGB', (512, 512), color='blue')
    draw = ImageDraw.Draw(img)
    # Add simple smooth shape
    draw.ellipse((100, 100, 400, 400), fill='red', outline='blue')
    return img


def debug_thresholds():
    engine = ForensicEngine()
    
    with open("debug_log.txt", "w") as f:
        # Helper to print to both
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log("\n--- DEBUGGING REAL-LIKE IMAGE (Random Noise) ---")
        real_img = create_real_like_image()
        res_real = engine.analyze(real_img)
        print_metrics(res_real, log)
        
        log("\n--- DEBUGGING AI-LIKE IMAGE (Smooth Shapes) ---")
        ai_img = create_ai_like_image()
        res_ai = engine.analyze(ai_img)
        print_metrics(res_ai, log)

def print_metrics(res, log_func):
    d = res['details']
    log_func(f"VERDICT: {res['forensicVote']} (Score: {res['forensicScore']})")
    
    log_func(f"[Pixel] ELA Max Diff: {d['pixel']['ela']['max_ela_diff']:.4f}")
    log_func(f"[Pixel] Noise Std:    {d['pixel']['noise']['std']:.4f}")
    log_func(f"[Freq]  FFT Energy:   {d['frequency']['fft']['high_freq_energy']:.4f}")
    log_func(f"[Freq]  DWT HH Ratio: {d['frequency']['dwt']['hh_ratio']:.6f}")
    
    if d['pixel']['ela']['max_ela_diff'] < 0.02: log_func("-> ELA votes AI (Too smooth)")
    elif d['pixel']['ela']['max_ela_diff'] > 0.3: log_func("-> ELA votes AI (Manipulated)")
    else: log_func("-> ELA votes Real")
    
    noise_val = d['pixel']['noise']['std']
    if noise_val < 2.0: log_func("-> Noise votes AI (Too smooth)")
    elif noise_val > 20.0: log_func("-> Noise votes AI (Too noisy)")
    else: log_func("-> Noise votes Real")
    
    fft_val = d['frequency']['fft']['high_freq_energy']
    if fft_val > 155: log_func("-> FFT votes AI (Artifacts)")
    else: log_func("-> FFT votes Real")
    
    dwt_val = d['frequency']['dwt']['hh_ratio']
    if dwt_val < 0.0001: log_func("-> DWT votes AI (Low Detail)")
    elif dwt_val < 0.0005: log_func("-> DWT votes AI (Med Detail)")
    else: log_func("-> DWT votes Real")


if __name__ == "__main__":
    debug_thresholds()
