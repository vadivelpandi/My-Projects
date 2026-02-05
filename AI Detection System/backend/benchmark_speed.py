import time
from PIL import Image
import numpy as np
import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add backend to path
sys.path.append(os.getcwd())

from model_manager import ModelManager
from forensic_engine import ForensicEngine
from metadata_engine import MetadataEngine

def create_dummy_image():
    return Image.fromarray(np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8))

def benchmark_sync():
    print("--- Starting Sync Benchmark ---")
    
    print("Initializing Engines...")
    mm = ModelManager()
    fe = ForensicEngine()
    me = MetadataEngine()
    
    # Wait for models to load (simulate startup)
    print("Waiting 5s for background loading...")
    time.sleep(5) 
    
    img = create_dummy_image()
    img_bytes = b"fake_bytes" # Metadata engine just needs bytes, content doesn't matter for speed test mostly
    
    start_total = time.time()
    
    print("Running ModelManager...")
    t1 = time.time()
    # mm.predict_full_suite(img) 
    # Use a try-catch for the suite in case models aren't fully loaded, but we want to time the logic
    try:
        mm.predict_full_suite(img)
    except Exception as e:
        print(f"MM Error: {e}")
    t2 = time.time()
    print(f"ModelManager Time: {t2 - t1:.4f}s")
    
    print("Running ForensicEngine...")
    fe.analyze(img)
    t3 = time.time()
    print(f"ForensicEngine Time: {t3 - t2:.4f}s")
    
    print("Running MetadataEngine...")
    # Metadata engine uses subprocess, so we need a real file really, but let's pass bytes
    # It might error out fast if invalid, but let's try to simulate overhead
    try:
        me.analyze(img_bytes) 
    except:
        pass
    t4 = time.time()
    print(f"MetadataEngine Time: {t4 - t3:.4f}s")
    
    print(f"Total Sequential Time: {t4 - start_total:.4f}s")
    print("-------------------------------")

def benchmark_async():
    print("--- Starting Async Benchmark (Simulated) ---")
    mm = ModelManager()
    fe = ForensicEngine()
    me = MetadataEngine()
    
    # Wait for loading
    time.sleep(5)
    
    img = create_dummy_image()
    img_bytes = b"fake_bytes"
    
    start_total = time.time()
    
    async def run_parallel():
        # Simulate main.py behavior
        t_ml = asyncio.to_thread(mm.predict_full_suite, img)
        t_fe = asyncio.to_thread(fe.analyze, img)
        # t_me = asyncio.to_thread(me.analyze, img_bytes) # Skip IO for now or mock it
        
        await asyncio.gather(t_ml, t_fe) #, t_me)
        
    asyncio.run(run_parallel())
    
    end_total = time.time()
    print(f"Total Parallel Time: {end_total - start_total:.4f}s")
    print("-------------------------------")

if __name__ == "__main__":
    benchmark_sync()
    benchmark_async()
