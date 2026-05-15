# Bovine Intelligence System

Streamlit app for bovine breed detection and classification.

## Run locally

1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   streamlit run app.py
   ```
3. Open the URL shown in the terminal (usually `http://localhost:8501`).

## GitHub setup

1. Initialize git (already done locally):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
2. Add your GitHub remote and push:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git branch -M main
   git push -u origin main
   ```

## Streamlit deployment

1. Go to https://streamlit.io/cloud and sign in with GitHub.
2. Create a new app and connect your GitHub repository.
3. Set the main branch and app file path to `app.py`.
4. Deploy the app.

> Note: If your model file is large, Streamlit Cloud may need extra time to install the repo. If the app fails due to file limits, move the model to an external storage or use Git LFS.
