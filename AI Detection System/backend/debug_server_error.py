
import asyncio
from fastapi import UploadFile
from main import app, analyze_content
import io

# Mock UploadFile
class MockFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content_type = "image/jpeg"
        self.file = io.BytesIO(content)
    async def read(self):
        self.file.seek(0)
        return self.file.read()


async def debug_run():
    print("Loading test image...")
    # Create valid dummy image content (using PIL to make it valid)
    from PIL import Image
    import numpy as np
    import time
    
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    content = buf.getvalue()
    
    mock_file = MockFile("test.jpg", content)
    
    # Access the global model_manager from main
    from main import model_manager
    
    print("Waiting for models to load...")
    # Wait max 60 seconds
    for _ in range(12):
        loaded = sum(1 for s in model_manager.loading_status.values() if s == "Ready")
        total = len(model_manager.model_names)
        print(f"Models loaded: {loaded}/{total}")
        if loaded > 0: # At least one model
            break
        await asyncio.sleep(5)
        
    print("Calling analyze_content...")
    start_time = time.time()
    try:
        # We need to manually trigger the dependency injection or just call the function
        # But analyze_content takes a UploadFile.
        result = await analyze_content(mock_file)
        duration = time.time() - start_time
        print(f"Success! took {duration:.2f} seconds")
        print("Result Classification:", result.get("classification"))
        print("Model Consensus:", result.get("modelConsensus"))
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn # Ensure uvicorn is imported if needed by main logic
    asyncio.run(debug_run())

