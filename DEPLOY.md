# Deployment Guide

This application uses a separate Frontend (Vercel) and Backend (Render).

> [!WARNING]
> **Gmail OAuth on Server**: Authentication requires a browser callback.
> Since Render servers are headless (no display), the standard "click to sync" flow will fail with `Browser not found` or similar errors if no token exists.
> **Workaround**: You must upload a valid `token.json` (generated locally) to your deployment.

## Part 1: Backend (Render)

1.  **Create Service on Render**:
    - Click **New +** -> **Web Service**.
    - Connect your repository.

2.  **Configuration**:
    - **Name**: `fusion-mail-api`
    - **Runtime**: `Python 3`
    - **Build Command**: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
    - **Start Command**: `gunicorn backend.app:app`

3.  **Environment Variables**:
    - Go to **Environment** in Render dashboard.
    - Add variables from your local `.env`:
        - `GOOGLE_CLIENT_ID`
        - `GOOGLE_CLIENT_SECRET`
        - `GOOGLE_PROJECT_ID`
        - `GOOGLE_TOKEN_JSON` (Paste the entire JSON string)

4.  **Deploy**:
    - Note the URL (e.g., `https://fusion-mail-api.onrender.com`). You will need this for the Frontend.

## Part 2: Frontend (Vercel)

1.  **Import Project**:
    - Go to [vercel.com/new](https://vercel.com/new).
    - Import the same repository.

2.  **Configuration**:
    - **Framework Preset**: Other (or just leave default).
    - **Project Settings**:
        - **Root Directory**: `.` (Default).
        - **Build Command**: (None).
        - **Output Directory**: (None).
    - *Note*: The included `vercel.json` automatically routes traffic to the `frontend/` folder, so no custom config is needed.

3.  **Environment Connectivity**:
    - The frontend automatically detects if it's running on `localhost`.
    - If running on Vercel, it connects to: `https://mail-app.onrender.com` (Hardcoded for this request).
    - *To change this in the future, edit `frontend/app.js` constant `API_BASE_URL`.*

4.  **Deploy**:
    - Click **Deploy**.

## Part 3: Critical Final Step (CORS)
- Because the Frontend (Vercel) and Backend (Render) are on different domains, the Backend must allow requests from Vercel.
- We have already updated `backend/app.py` to enable **CORS**.
- **ACTION**: You MUST redeploy your Render Backend to pick up this change, otherwise the Vercel Frontend will get "Network Error" when trying to fetch data.
