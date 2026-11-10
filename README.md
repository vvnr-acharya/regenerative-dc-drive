---
title: Regenerative DC Drive
emoji: ⚡
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Regenerative DC Drive

FastAPI backend and web-based digital twin for a regenerative DC drive. The app includes a physics simulation, live telemetry over WebSockets, and trained ML estimates.

## Deploy as a Hugging Face Space

1. Create a new **public Space** using the **Docker** SDK and free CPU hardware.
2. Push this repository's files to the Space's Git repository.
3. In the Space **Settings → Variables and secrets**, add:
   - `ADMIN_USERNAME`
   - `ADMIN_PASSWORD` (use a new, unique password)
4. Wait for the Space to build, then open its `https://<space-name>.hf.space` URL. The admin portal is at `/admin`.

The credentials must be stored as Space secrets, not committed to the repository. Public Spaces expose their source code. The bundled model artifacts and dataset are simulation-derived and are not validated for vehicle safety.
