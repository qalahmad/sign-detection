# 🚀 Deploying VocalHands to Streamlit Cloud

## Prerequisites

1. **GitHub Account** - [Sign up here](https://github.com)
2. **Streamlit Cloud Account** - [Sign up here](https://streamlit.io/cloud) (free with GitHub)
3. **Trained Model** - Make sure `models/knn_sign_model.pkl` exists

---

## Step 1: Prepare Your Repository

### Create a new GitHub repository

1. Go to [github.com/new](https://github.com/new)
2. Name it `VocalHands` (or any name you prefer)
3. Make it **Public** (required for free Streamlit Cloud)
4. Don't initialize with README (we have one)

### Initialize Git and Push

Open terminal in `VocalHands` folder:

```bash
# Initialize git
git init

# Add files (excluding large data folders)
git add streamlit_app.py
git add requirements_streamlit.txt
git add models/knn_sign_model.pkl
git add .streamlit/config.toml
git add .gitignore
git add README.md
git add utils/

# Commit
git commit -m "Initial commit - VocalHands Sign Language Detection"

# Add your GitHub repo as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/VocalHands.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Click **"New app"**

3. Fill in the details:
   - **Repository**: `YOUR_USERNAME/VocalHands`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`

4. **Advanced settings** (click to expand):
   - **Python version**: 3.10
   - **Requirements file**: `requirements_streamlit.txt`

5. Click **"Deploy!"**

---

## Step 3: Wait for Deployment

- Streamlit will install dependencies and start your app
- This takes 2-5 minutes the first time
- You'll get a URL like: `https://your-app-name.streamlit.app`

---

## Troubleshooting

### "Model not found" error
Make sure `models/knn_sign_model.pkl` is committed to GitHub:
```bash
git add -f models/knn_sign_model.pkl
git commit -m "Add trained model"
git push
```

### "Module not found" error
Check that `requirements_streamlit.txt` is in the repo root and selected in Streamlit settings.

### Camera not working
- Allow camera permissions in your browser
- Use HTTPS (Streamlit Cloud provides this automatically)
- Try a different browser if issues persist

---

## Files Required for Deployment

```
VocalHands/
├── streamlit_app.py           # Main Streamlit app
├── requirements_streamlit.txt # Dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── models/
│   └── knn_sign_model.pkl    # Trained model (REQUIRED!)
└── utils/
    ├── __init__.py
    └── hand_detector.py      # Hand detection utilities
```

---

## Updating Your App

After making changes locally:

```bash
git add .
git commit -m "Update app"
git push
```

Streamlit Cloud will automatically redeploy!

---

## Custom Domain (Optional)

1. Go to your app settings on Streamlit Cloud
2. Click "Custom subdomain"
3. Enter your preferred name (e.g., `vocalhands`)
4. Your app will be at: `https://vocalhands.streamlit.app`

---

## Need Help?

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud Guide](https://docs.streamlit.io/streamlit-community-cloud)
- [MediaPipe Docs](https://google.github.io/mediapipe/)

---

🎉 **Congratulations!** Your VocalHands app is now live on the web!
