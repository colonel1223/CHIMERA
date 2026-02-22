#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# CHIMERA — GitHub Deployment Script
# Run these commands in your local terminal (requires git + gh CLI)
# ═══════════════════════════════════════════════════════════════

# OPTION A: Using GitHub CLI (fastest — install: brew install gh)
# ─────────────────────────────────────────────────────────────

# 1. Navigate to your chimera folder (wherever you unzipped/downloaded)
cd ~/Downloads/chimera-repo  # ← adjust path as needed

# 2. Initialize git repo
git init
git add -A
git commit -m "🜂 CHIMERA: Confabulation Hazard Index — AI Hallucination as Modern Alchemy

Formal mathematical framework proving hallucination is structural, not fixable.
Includes: Impossibility Theorem, Jungian Shadow analysis, alchemical topology,
interactive 9-panel React model, and full research manuscript."

# 3. Create GitHub repo and push (public)
gh repo create CHIMERA --public --source=. --remote=origin --push

# Done. Your repo is live.

# ─────────────────────────────────────────────────────────────
# OPTION B: Manual (if you don't have gh CLI)
# ─────────────────────────────────────────────────────────────

# 1. Go to https://github.com/new
# 2. Name: CHIMERA | Public | No README (we have one) | Create
# 3. Then run:

cd ~/Downloads/chimera-repo  # ← adjust path
git init
git add -A
git commit -m "🜂 CHIMERA: Confabulation Hazard Index — AI Hallucination as Modern Alchemy"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CHIMERA.git  # ← replace
git push -u origin main

# ═══════════════════════════════════════════════════════════════
