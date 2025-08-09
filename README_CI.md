# FinGPT Flask: Compose + PR CI Drop-in

## What this bundle gives you
- A `Dockerfile` for Flask/Gunicorn.
- `docker-compose.yml` with a healthcheck on `/health`.
- GitHub Actions workflow that runs on Pull Requests:
  - black/flake8
  - pytest (if `tests/` exists)
  - builds the Docker image
  - spins container and smoke-checks `/health`

## How to integrate
1. Copy these files into your repo root (keep the folder structure):
   - `Dockerfile`
   - `docker-compose.yml`
   - `.github/workflows/pr-ci.yml`
   - `.env.example`
   - `tests/test_smoke.py` (optional but recommended)
2. Ensure your Flask entrypoint matches the CMD in `Dockerfile` (`app:app` by default).
3. Implement a `/health` route that returns 200 OK.
4. In GitHub, set any required secrets in **Settings → Secrets and variables → Actions**.
5. Open a Pull Request to see the CI run.

## Local run
```
cp .env.example .env
docker compose up --build
# visit http://localhost:8000
```