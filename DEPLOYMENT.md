
# ☁️ How to Deploy E-Commerce Smart Pricing Engine

This guide will help you deploy your Flask application to the web so others can use it. We recommend using **Render** because it is free and easy to set up for Flask apps.

## ✅ Prerequisites

1.  **GitHub Account:** You must have your code pushed to GitHub (which we just did!).
2.  **Render Account:** Sign up for free at [render.com](https://render.com/).

## 🚀 Deployment Steps

1.  **Log in to Render** and click the **"New +"** button.
2.  Select **"Web Service"**.
3.  **Connect GitHub:**
    *   Click "Build and deploy from a Git repository".
    *   Connect your GitHub account if you haven't already.
    *   Search for and select: `ecommerce-smart-pricing-engine`.
4.  **Configure Settings:**
    *   **Name:** Choose a unique name (e.g., `my-pricing-app`).
    *   **Region:** Select the one closest to you (e.g., Singapore or Frankfurt).
    *   **Branch:** `main`.
    *   **Runtime:** `Python 3`.
    *   **Build Command:** `pip install -r requirements.txt`.
    *   **Start Command:** `gunicorn app:app`.
    *   **Plan:** Select **Free**.
5.  **Click "Create Web Service"**.

## ⏳ What Happens Next?

*   Render will start building your app. You can verify the logs in the dashboard.
*   It may take a few minutes to install dependencies (scikit-learn, pandas, etc.).
*   Once finished, you will see a green **"Live"** badge.
*   Your app will be accessible at: `https://<your-app-name>.onrender.com`.

## 🔗 How to Share with Others

Simply send them the URL! (e.g., `https://my-pricing-app.onrender.com`). They can access it from any device without installing anything.
