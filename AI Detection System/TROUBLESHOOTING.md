# Troubleshooting Guide

Quick solutions to common deployment issues.

## 🔍 Quick Diagnostics

Run these commands to check system status:

```bash
# Check if Docker is running
docker --version

# Check container status
docker-compose ps

# View logs
docker-compose logs

# Check specific service logs
docker-compose logs backend
docker-compose logs frontend
```

---

## 🚨 Common Issues & Solutions

### 1. Port Already in Use

**Error:** `Bind for 0.0.0.0:80 failed: port is already allocated`

**Solution:**

**Option A - Find and stop the conflicting process:**
```bash
# Windows
netstat -ano | findstr :80
taskkill /PID <PID_NUMBER> /F

# Linux/Mac
lsof -i :80
kill -9 <PID>
```

**Option B - Change the port:**
Edit `docker-compose.yml`:
```yaml
services:
  frontend:
    ports:
      - "3000:80"  # Changed from 80 to 3000
  backend:
    ports:
      - "8001:8000"  # Changed from 8000 to 8001
```

Then access at http://localhost:3000

---

### 2. Docker Not Running

**Error:** `Cannot connect to the Docker daemon`

**Solution:**
1. Start Docker Desktop application
2. Wait for it to fully start (whale icon in system tray)
3. Try the command again

**Windows:** Check if Docker Desktop is running in system tray
**Mac:** Check menu bar for Docker icon
**Linux:** Run `sudo systemctl start docker`

---

### 3. Out of Memory / Container Crashes

**Error:** Container exits with code 137 or 143

**Solution:**

**Increase Docker Memory:**
1. Open Docker Desktop
2. Settings → Resources
3. Increase Memory to 8GB minimum (16GB recommended)
4. Click "Apply & Restart"

**Check current usage:**
```bash
docker stats
```

**Use production config with limits:**
```bash
docker-compose -f production.yml up -d
```

---

### 4. Models Not Loading

**Error:** Backend logs show model download failures

**Solution:**

**Check internet connection:**
```bash
# Test Hugging Face connectivity
curl -I https://huggingface.co
```

**Increase timeout and retry:**
1. Wait longer (first download can take 10-15 minutes)
2. Check logs: `docker-compose logs -f backend`
3. Look for "Initializing Model Manager..." message

**Manual model download (if needed):**
```bash
docker-compose exec backend python -c "from transformers import pipeline; pipeline('image-classification', model='umm-maybe/AI-image-detector')"
```

---

### 5. CORS Errors in Browser

**Error:** `Access to fetch at 'http://localhost:8000/analyze' from origin 'http://localhost' has been blocked by CORS policy`

**Solution:**

**Verify backend CORS settings** in `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Check nginx proxy** in `frontend/nginx.conf`:
```nginx
location /analyze {
    proxy_pass http://backend:8000;
    # ... other settings
}
```

**Rebuild if changed:**
```bash
docker-compose down
docker-compose up --build -d
```

---

### 6. File Upload Fails

**Error:** `413 Request Entity Too Large` or upload hangs

**Solution:**

**Increase nginx upload limit** in `frontend/nginx.conf`:
```nginx
client_max_body_size 100M;  # Increase if needed
```

**Increase timeouts:**
```nginx
proxy_connect_timeout 300s;
proxy_send_timeout 300s;
proxy_read_timeout 300s;
```

**Rebuild frontend:**
```bash
docker-compose up --build frontend
```

---

### 7. Frontend Shows Blank Page

**Error:** White screen or "Cannot GET /"

**Solution:**

**Check if container is running:**
```bash
docker-compose ps
```

**Check frontend logs:**
```bash
docker-compose logs frontend
```

**Verify build succeeded:**
```bash
docker-compose exec frontend ls -la /usr/share/nginx/html
```

**Rebuild frontend:**
```bash
docker-compose down
docker-compose up --build -d
```

---

### 8. Backend API Not Responding

**Error:** `Failed to fetch` or connection refused

**Solution:**

**Check backend status:**
```bash
docker-compose ps backend
```

**Check backend logs:**
```bash
docker-compose logs backend
```

**Test backend directly:**
```bash
curl http://localhost:8000/
```

**Restart backend:**
```bash
docker-compose restart backend
```

---

### 9. Slow Performance / Analysis Takes Too Long

**Issue:** Analysis takes more than 30 seconds

**Solution:**

**Check resource allocation:**
```bash
docker stats
```

**Increase resources in Docker Desktop:**
- CPU: 4+ cores
- Memory: 8GB+ (16GB recommended)

**Use GPU acceleration (if available):**
Edit `docker-compose.yml`:
```yaml
backend:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Reduce number of models** (edit `backend/model_manager.py`):
- Comment out some models to speed up inference

---

### 10. Build Fails

**Error:** Build errors during `docker-compose up --build`

**Solution:**

**Clear Docker cache:**
```bash
docker-compose down
docker system prune -a
docker-compose up --build
```

**Check disk space:**
```bash
# Windows
dir
# Linux/Mac
df -h
```

**Check for syntax errors:**
- Review recent code changes
- Check Dockerfile syntax
- Verify requirements.txt format

---

## 🔧 Advanced Troubleshooting

### View Detailed Container Info
```bash
docker inspect <container_name>
```

### Access Container Shell
```bash
# Backend
docker-compose exec backend bash

# Frontend
docker-compose exec frontend sh
```

### Check Network Connectivity
```bash
# Test backend from frontend container
docker-compose exec frontend wget -O- http://backend:8000/
```

### Reset Everything
```bash
# Nuclear option - removes everything
docker-compose down -v
docker system prune -a
docker volume prune
docker-compose up --build
```

---

## 📊 Health Check Commands

```bash
# Quick health check
curl http://localhost:8000/
curl http://localhost/

# Detailed status
docker-compose ps
docker-compose logs --tail=50

# Resource usage
docker stats --no-stream

# Network inspection
docker network ls
docker network inspect <network_name>
```

---

## 🆘 Still Having Issues?

### Collect Debug Information

1. **System info:**
   ```bash
   docker version
   docker-compose version
   ```

2. **Container status:**
   ```bash
   docker-compose ps
   ```

3. **Recent logs:**
   ```bash
   docker-compose logs --tail=100 > debug_logs.txt
   ```

4. **Resource usage:**
   ```bash
   docker stats --no-stream > resource_usage.txt
   ```

### Check Documentation
- DEPLOYMENT.md - Comprehensive deployment guide
- SYSTEM_DETAILS.md - System architecture
- README_DEPLOY.md - Quick start guide

### Common Gotchas
- First run takes 10-15 minutes (model downloads)
- Windows: Use PowerShell or CMD, not Git Bash
- Mac M1/M2: Some models may need ARM-specific builds
- Linux: May need `sudo` for Docker commands

---

## 📝 Logging Best Practices

### Enable Debug Logging

**Backend** - Edit `backend/main.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Frontend** - Check browser console (F12)

### Save Logs for Analysis
```bash
docker-compose logs > full_logs.txt
docker-compose logs backend > backend_logs.txt
docker-compose logs frontend > frontend_logs.txt
```

---

## ✅ Verification Checklist

After fixing issues, verify:
- [ ] Containers are running: `docker-compose ps`
- [ ] Backend responds: `curl http://localhost:8000/`
- [ ] Frontend loads: Open http://localhost in browser
- [ ] File upload works: Test with sample image
- [ ] Analysis completes: Check results display
- [ ] No errors in logs: `docker-compose logs`

---

**Need more help?** Check the other documentation files or review the logs carefully for specific error messages.
