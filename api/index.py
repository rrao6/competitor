"""
Vercel Serverless Entry Point for Competitor Monitor Dashboard.

This module adapts the Flask app for Vercel's serverless environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Set up path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the Flask app
from dashboard.app import app

# Vercel expects an 'app' variable
# The app is already configured in dashboard/app.py
