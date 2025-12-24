# Deployment Guide to Render

This application uses Flask and TensorFlow/PyTorch.

> [!WARNING]
> **Gmail OAuth on Server**: Authentication requires a browser callback.
> Since Render servers are headless (no display), the standard "click to sync" flow will fail with `Browser not found` or similar errors if no token exists.
> **Workaround**: You must upload a valid `token.json` (generated locally) to your deployment.

## Prerequisites
1.  **Git Repository**: Push this code to GitHub/GitLab.
2.  **Render Account**: Sign up at [render.com](https://render.com).

## Deployment Steps

1.  **Create Service on Render**:
    - Click **New +** -> **Web Service**.
    - Connect your repository.

2.  **Configuration**:
    - **Name**: `fusion-mail-app` (or similar)
    - **Runtime**: `Python 3`
    - **Build Command**: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
    - **Start Command**: `gunicorn backend.app:app` (Note: we updated Procfile to `cd backend && gunicorn app:app`, but Render might need manual Start Command if it ignores Procfile. Try `cd backend && gunicorn app:app`).

3.  **Environment Variables / Secrets**:
    - Go to **Environment** in Render dashboard.
    - Add the following variables (copy values from your local `.env`):
        - `GOOGLE_CLIENT_ID`
        - `GOOGLE_CLIENT_SECRET`
        - `GOOGLE_PROJECT_ID`
        - `GOOGLE_TOKEN_JSON` (Paste the entire JSON string)
    
    - **Alternative**: You can still upload your `.env` file as a "Secret File" named `.env` if you prefer that over individual variables.

4.  **Deploy**:
    - Click **Create Web Service**.

## Notes
- The `fusion_model.pth` is large. If you commit it to Git (requires LFS), Render will pull it.
- If not committed, the app will use random weights (or you can add `python ml/train_pipeline.py` to the Build Command to train a fresh one).
