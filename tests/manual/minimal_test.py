#!/usr/bin/env python3
"""Minimal test to isolate when 'No segments produced' messages occur"""

print("🔍 Starting minimal test...")

print("1. Importing Config...")
from src.utils.config import Config

print("2. Loading Config...")
config = Config()

print("3. Importing SubtitleGenerator...")
from src.subtitle.subtitle_generator import SubtitleGenerator

print("4. Creating SubtitleGenerator...")
generator = SubtitleGenerator(config)

print("5. Importing MainWindow...")
from src.ui.main_window import MainWindow

print("6. Creating MainWindow (without running)...")
# Don't actually run the GUI, just create it
import tkinter as tk
root = tk.Tk()
root.withdraw()  # Hide the window

app = MainWindow(config)
print("7. MainWindow created successfully")

print("8. Cleaning up...")
root.destroy()

print("✅ Minimal test completed - if 'No segments produced' appeared above, we know which step caused it")