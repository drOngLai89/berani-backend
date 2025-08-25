# Restore Backend (Render + Flask)

## Option A: Render Blueprint (recommended)
1) In Render, New + → Blueprint
2) Select this repo and render.yaml
3) Set OPENAI_API_KEY env var (leave blank if you want stub replies)
4) Deploy — service URL should become https://berani-backend.onrender.com or similar

## Option B: Manual Web Service
- Repo: drOngLai89/berani-backend (branch main)
- Build command: pip install -r requirements.txt
- Start command: gunicorn server:app -b 0.0.0.0:$PORT
- Env var: OPENAI_API_KEY (optional for live AI)

## Health check
./scripts/healthcheck.sh
