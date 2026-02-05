# Getting Started - AI Content Detector Deployment

## 🎯 Goal
Deploy your AI Content Detector application in under 10 minutes.

## ⚡ Prerequisites (5 minutes)

1. **Install Docker Desktop**
   - Windows/Mac: Download from https://www.docker.com/products/docker-desktop
   - Linux: Follow instructions at https://docs.docker.com/engine/install/
   - After installation, start Docker Desktop

2. **Verify Installation**
   Open terminal/command prompt and run:
   ```bash
   docker --version
   docker-compose --version
   ```
   Both should show version numbers.

## 🚀 Deploy in 3 Steps (5 minutes)

### Step 1: Open Terminal
- **Windows**: Open Command Prompt or PowerShell
- **Mac/Linux**: Open Terminal
- Navigate to your project directory

### Step 2: Run Deployment Script

**Windows:**
```cmd
deploy.bat
```

**Mac/Linux:**
```bash
chmod +x deploy.sh
./deploy.sh
```

### Step 3: Choose Option 1
- Select option 1: "Build and start containers"
- Wait for the build to complete (5-10 minutes first time)
- You'll see "Deployment complete!" when ready

## ✅ Verify It's Working

1. **Open your browser**
2. **Go to:** http://localhost
3. **You should see:** The AI Content Detector interface
4. **Test it:** Upload an image and click "Analyze Content"

## 🎉 That's It!

Your application is now running:
- **Frontend UI**: http://localhost
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🛠️ Common Commands

```bash
# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Restart after code changes
docker-compose up --build -d

# Check container status
docker-compose ps
```

## 🆘 Something Wrong?

### Port Already in Use?
Edit `docker-compose.yml` and change the port numbers:
```yaml
frontend:
  ports:
    - "3000:80"  # Changed from 80 to 3000
```

### Out of Memory?
1. Open Docker Desktop
2. Go to Settings → Resources
3. Increase Memory to at least 8GB
4. Click "Apply & Restart"

### Still Having Issues?
1. Check logs: `docker-compose logs -f`
2. Run health check: `health-check.bat` (Windows)
3. See DEPLOYMENT.md for detailed troubleshooting

## 📚 Next Steps

- **For Production**: Read DEPLOYMENT.md
- **For Cloud Deployment**: See cloud deployment section in DEPLOYMENT.md
- **For Customization**: Check SYSTEM_DETAILS.md

## 💡 Tips

- First run downloads ML models (~2-5GB) - be patient
- Models are cached, subsequent starts are much faster
- Use `deploy.bat` or `deploy.sh` for easy management
- Check `docker-compose logs` if something seems wrong

---

**Need more help?** Check out:
- README_DEPLOY.md - Quick deployment guide
- DEPLOYMENT.md - Comprehensive deployment documentation
- DEPLOYMENT_CHECKLIST.md - Pre-deployment checklist
