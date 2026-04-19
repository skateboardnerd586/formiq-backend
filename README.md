---
title: FormIQ Backend
emoji: 🏋️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# FormIQ Backend

Pose detection and lifting form analysis API. POST a video to `/api/analyze/deadlift` or `/api/analyze/overhead-squat`, get back per-rep joint angles and Claude-generated coaching feedback.

## Required Space secrets

Set these under **Settings → Variables and secrets** on the Space:

- `ANTHROPIC_API_KEY` — your Anthropic API key
- `ALLOWED_ORIGINS` — comma-separated list, e.g. `https://formiq-frontend.vercel.app,http://localhost:3000`
- `ALLOWED_ORIGIN_REGEX` *(optional)* — regex for Vercel preview URLs
