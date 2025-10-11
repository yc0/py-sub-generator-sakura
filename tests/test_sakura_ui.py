#!/usr/bin/env uv run python
"""
Test script for SakuraLLM UI integration.
Run with: uv run python test_sakura_ui.py
"""

import sys
from pathlib import Path
import tkinter as tk

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config import Config
from src.ui.components.settings_dialog import SettingsDialog


def test_ui():
    """Test the SakuraLLM UI integration."""
    print("🌸 Testing SakuraLLM UI Integration")
    
    # Create root window
    root = tk.Tk()
    root.title("SakuraLLM UI Test")
    root.geometry("300x200")
    
    # Create config
    config = Config()
    
    # Create a button to open settings
    def open_settings():
        dialog = SettingsDialog(root, config)
        result = dialog.show()
        if result:
            print("✅ Settings saved successfully!")
            
            # Show SakuraLLM status
            sakura_config = config.get_sakura_config()
            print(f"SakuraLLM Enabled: {config.is_sakura_enabled()}")
            if config.is_sakura_enabled():
                print(f"Model: {sakura_config.get('model_name')}")
                print(f"Device: {sakura_config.get('device')}")
                print(f"Temperature: {sakura_config.get('temperature')}")
        else:
            print("❌ Settings cancelled")
    
    # Add UI elements
    tk.Label(root, text="🌸 SakuraLLM UI Test", font=("Arial", 16)).pack(pady=20)
    tk.Button(root, text="Open Settings", command=open_settings, font=("Arial", 12)).pack(pady=10)
    
    # Instructions
    instructions = tk.Text(root, height=4, wrap=tk.WORD)
    instructions.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    instructions.insert(tk.END, 
        "Click 'Open Settings' to test the SakuraLLM configuration UI.\n\n"
        "Look for:\n"
        "• 🌸 SakuraLLM tab\n"
        "• MPS device option\n"
        "• Model selection with VRAM info"
    )
    instructions.config(state=tk.DISABLED)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    test_ui()