# AI Content Detector

A full-stack application that detects AI-generated images and videos using a multi-model ensemble approach.

## 🚀 Quick Deploy

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed
- 8GB+ RAM available
- 10GB free disk space

### Deploy Now

**Windows:**
```cmd
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

Select option 1 and wait for deployment to complete (~10 minutes first time).

### Access Application
- **Frontend**: http://localhost
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📋 Features

- **Multi-Model Ensemble**: Uses 11 pre-trained AI detection models
- **Image Analysis**: Detects AI-generated images with high accuracy
- **Video Analysis**: Frame-by-frame analysis of video content
- **Real-time Results**: Instant feedback with confidence scores
- **Model Consensus**: Shows individual model votes and agreement levels
- **Forensic Analysis**: ELA, noise pattern, and frequency analysis
- **Interactive UI**: Modern, responsive React interface

## 🏗️ Architecture

- **Frontend**: React + Vite + Tailwind CSS
- **Backend**: FastAPI + PyTorch + Transformers
- **Deployment**: Docker + Docker Compose
- **Web Server**: Nginx (reverse proxy)

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | 10-minute quick start guide |
| [README_DEPLOY.md](README_DEPLOY.md) | Quick deployment reference |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Comprehensive deployment guide |
| [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md) | Deployment setup overview |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |
| [SYSTEM_DETAILS.md](SYSTEM_DETAILS.md) | System architecture details |
| [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) | Pre-deployment checklist |

## 🛠️ Development

### Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild
docker-compose up --build -d
```

## 🌐 Deployment Options

### Local (Docker)
```bash
docker-compose up -d
```

### Production (Docker with limits)
```bash
docker-compose -f production.yml up -d
```

### Cloud Platforms
- AWS EC2
- DigitalOcean Droplet
- Google Cloud Run
- Azure Container Instances
- Heroku

See [DEPLOYMENT.md](DEPLOYMENT.md) for platform-specific instructions.

## 🔧 Configuration

### Change Ports
Edit `docker-compose.yml`:
```yaml
services:
  frontend:
    ports:
      - "3000:80"  # Change first number
  backend:
    ports:
      - "8001:8000"  # Change first number
```

### Environment Variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 🧪 Testing

1. Start the application
2. Open http://localhost
3. Upload a test image
4. Click "Analyze Content"
5. Verify results display correctly

## 📊 ML Models Used

The system uses 11 pre-trained models:
1. umm-maybe/AI-image-detector
2. prithivMLmods/Deep-Fake-Detector-v2-Model
3. prithivMLmods/deepfake-detector-model-v1
4. Ateeqq/ai-vs-human-image-detector
5. jacoballessio/ai-image-detect-distilled
6. Hemg/AI-VS-REAL-IMAGE-DETECTION
7. dima806/ai_vs_real_image_detection
8. Organika/sdxl-detector
9. Wvolf/ViT_Deepfake_Detection
10. NHNDQ/vit-base-patch16-224-finetuned-deepfake
11. Nahrawy/AIorNot

Models are automatically downloaded on first run.

## 🐛 Troubleshooting

### Common Issues

**Port already in use:**
- Change ports in docker-compose.yml

**Out of memory:**
- Increase Docker memory limit in Docker Desktop settings

**Models not loading:**
- Check internet connection
- Wait longer (first download takes time)

**Full troubleshooting guide:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 📝 Project Structure

```
.
├── backend/                 # FastAPI backend
│   ├── main.py             # API endpoints
│   ├── model_manager.py    # ML model management
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container config
├── frontend/               # React frontend
│   ├── src/
│   │   ├── App.jsx        # Main component
│   │   └── ...
│   ├── nginx.conf         # Nginx configuration
│   ├── Dockerfile         # Frontend container config
│   └── package.json       # Node dependencies
├── docker-compose.yml     # Main orchestration
├── production.yml         # Production config
├── deploy.bat            # Windows deployment script
├── deploy.sh             # Linux/Mac deployment script
└── DOCUMENTATION/        # Project documentation
```

## 🔒 Security Considerations

For production deployment:
- Enable HTTPS/SSL
- Implement rate limiting
- Add authentication
- Validate file uploads
- Use environment variables for secrets
- Configure firewall rules

See [DEPLOYMENT.md](DEPLOYMENT.md) for security best practices.

## 📈 Performance

- **First run**: 10-15 minutes (model downloads)
- **Subsequent runs**: < 1 minute startup
- **Image analysis**: 1-3 seconds
- **Video analysis**: 3-8 seconds (5 frames)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license here]

## 👥 Authors

[Add author information]

## 🙏 Acknowledgments

- Hugging Face for pre-trained models
- FastAPI framework
- React and Vite teams
- Docker community

## 📞 Support

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Review [DEPLOYMENT.md](DEPLOYMENT.md)
- Check logs: `docker-compose logs -f`

---

**Ready to deploy?** Run `deploy.bat` (Windows) or `./deploy.sh` (Linux/Mac) to get started!
