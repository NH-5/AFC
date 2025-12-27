#!/usr/bin/env python3
"""
Run the web frontend

Usage:
    python run_web.py

This starts the Flask server and opens the web interface.
"""

import webbrowser
import threading
import time
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import app

def open_browser():
    """Open browser after server starts"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 50)
    print("AI Face Detection - Web Frontend")
    print("=" * 50)
    print()
    print("Starting server...")

    # Start browser in background
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
