"""Settings dialog for configuring application parameters."""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any
import logging

from ...utils.config import Config

logger = logging.getLogger(__name__)


class SettingsDialog:
    """Dialog for configuring application settings."""
    
    def __init__(self, parent: tk.Tk, config: Config):
        """Initialize settings dialog.
        
        Args:
            parent: Parent window
            config: Application configuration object
        """
        self.parent = parent
        self.config = config
        self.result = False
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("500x600")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center on parent
        self.center_on_parent()
        
        # Create widgets
        self.create_widgets()
        
        # Load current settings
        self.load_settings()
    
    def center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def create_widgets(self):
        """Create dialog widgets."""
        # Main frame with scrollbar
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # ASR Settings Tab
        self.create_asr_tab(notebook)
        
        # Translation Settings Tab
        self.create_translation_tab(notebook)
        
        # Output Settings Tab
        self.create_output_tab(notebook)
        
        # UI Settings Tab
        self.create_ui_tab(notebook)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Buttons
        ttk.Button(
            button_frame,
            text="OK",
            command=self.ok_clicked
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_clicked
        ).pack(side=tk.RIGHT)
        
        ttk.Button(
            button_frame,
            text="Reset to Default",
            command=self.reset_clicked
        ).pack(side=tk.LEFT)
    
    def create_asr_tab(self, notebook):
        """Create ASR settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="ASR")
        
        # Model selection
        ttk.Label(frame, text="Whisper Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.asr_model_var = tk.StringVar()
        model_combo = ttk.Combobox(
            frame,
            textvariable=self.asr_model_var,
            values=[
                "openai/whisper-tiny",
                "openai/whisper-base", 
                "openai/whisper-small",
                "openai/whisper-medium",
                "openai/whisper-large-v2",
                "openai/whisper-large-v3"
            ],
            state="readonly"
        )
        model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Device selection
        ttk.Label(frame, text="Device:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.asr_device_var = tk.StringVar()
        device_combo = ttk.Combobox(
            frame,
            textvariable=self.asr_device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly"
        )
        device_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Batch size
        ttk.Label(frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.asr_batch_var = tk.IntVar()
        batch_spin = tk.Spinbox(
            frame,
            from_=1,
            to=8,
            textvariable=self.asr_batch_var,
            width=10
        )
        batch_spin.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Chunk length
        ttk.Label(frame, text="Chunk Length (seconds):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.asr_chunk_var = tk.IntVar()
        chunk_spin = tk.Spinbox(
            frame,
            from_=10,
            to=60,
            textvariable=self.asr_chunk_var,
            width=10
        )
        chunk_spin.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Configure column weights
        frame.columnconfigure(1, weight=1)
    
    def create_translation_tab(self, notebook):
        """Create translation settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Translation")
        
        # Japanese to English model
        ttk.Label(frame, text="Japanese→English Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ja_en_model_var = tk.StringVar()
        ja_en_entry = ttk.Entry(frame, textvariable=self.ja_en_model_var, width=40)
        ja_en_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # English to Chinese model
        ttk.Label(frame, text="English→Chinese Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.en_zh_model_var = tk.StringVar()
        en_zh_entry = ttk.Entry(frame, textvariable=self.en_zh_model_var, width=40)
        en_zh_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Translation device
        ttk.Label(frame, text="Device:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.trans_device_var = tk.StringVar()
        trans_device_combo = ttk.Combobox(
            frame,
            textvariable=self.trans_device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly"
        )
        trans_device_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Translation batch size
        ttk.Label(frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.trans_batch_var = tk.IntVar()
        trans_batch_spin = tk.Spinbox(
            frame,
            from_=1,
            to=32,
            textvariable=self.trans_batch_var,
            width=10
        )
        trans_batch_spin.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Max length
        ttk.Label(frame, text="Max Sequence Length:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.trans_max_length_var = tk.IntVar()
        max_length_spin = tk.Spinbox(
            frame,
            from_=128,
            to=1024,
            increment=64,
            textvariable=self.trans_max_length_var,
            width=10
        )
        max_length_spin.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        frame.columnconfigure(1, weight=1)
    
    def create_output_tab(self, notebook):
        """Create output settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Output")
        
        # Default format
        ttk.Label(frame, text="Default Format:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.output_format_var = tk.StringVar()
        format_combo = ttk.Combobox(
            frame,
            textvariable=self.output_format_var,
            values=["srt", "vtt", "ass"],
            state="readonly"
        )
        format_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Include confidence
        self.include_confidence_var = tk.BooleanVar()
        confidence_check = ttk.Checkbutton(
            frame,
            text="Include confidence scores",
            variable=self.include_confidence_var
        )
        confidence_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Output directory
        ttk.Label(frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        output_dir_entry = ttk.Entry(frame, textvariable=self.output_dir_var, width=40)
        output_dir_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Temp directory
        ttk.Label(frame, text="Temp Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.temp_dir_var = tk.StringVar()
        temp_dir_entry = ttk.Entry(frame, textvariable=self.temp_dir_var, width=40)
        temp_dir_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        frame.columnconfigure(1, weight=1)
    
    def create_ui_tab(self, notebook):
        """Create UI settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="UI")
        
        # Window size
        ttk.Label(frame, text="Window Size:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.window_size_var = tk.StringVar()
        size_combo = ttk.Combobox(
            frame,
            textvariable=self.window_size_var,
            values=["800x600", "1000x700", "1200x800", "1400x900"],
            state="readonly"
        )
        size_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Theme
        ttk.Label(frame, text="Theme:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.theme_var = tk.StringVar()
        theme_combo = ttk.Combobox(
            frame,
            textvariable=self.theme_var,
            values=["default", "clam", "alt", "classic"],
            state="readonly"
        )
        theme_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Progress update interval
        ttk.Label(frame, text="Progress Update Interval (ms):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.progress_interval_var = tk.IntVar()
        interval_spin = tk.Spinbox(
            frame,
            from_=50,
            to=500,
            increment=50,
            textvariable=self.progress_interval_var,
            width=10
        )
        interval_spin.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        frame.columnconfigure(1, weight=1)
    
    def load_settings(self):
        """Load current settings into form."""
        try:
            # ASR settings
            asr_config = self.config.get_asr_config()
            self.asr_model_var.set(asr_config.get("model_name", "openai/whisper-large-v3"))
            self.asr_device_var.set(asr_config.get("device", "auto"))
            self.asr_batch_var.set(asr_config.get("batch_size", 1))
            self.asr_chunk_var.set(asr_config.get("chunk_length", 30))
            
            # Translation settings
            trans_config = self.config.get_translation_config()
            self.ja_en_model_var.set(trans_config.get("ja_to_en_model", "Helsinki-NLP/opus-mt-ja-en"))
            self.en_zh_model_var.set(trans_config.get("en_to_zh_model", "Helsinki-NLP/opus-mt-en-zh"))
            self.trans_device_var.set(trans_config.get("device", "auto"))
            self.trans_batch_var.set(trans_config.get("batch_size", 8))
            self.trans_max_length_var.set(trans_config.get("max_length", 512))
            
            # Output settings
            output_config = self.config.get_output_config()
            self.output_format_var.set(output_config.get("default_format", "srt"))
            self.include_confidence_var.set(output_config.get("include_confidence", True))
            self.output_dir_var.set(output_config.get("output_directory", "outputs"))
            self.temp_dir_var.set(output_config.get("temp_directory", "temp"))
            
            # UI settings
            ui_config = self.config.get_ui_config()
            self.window_size_var.set(ui_config.get("window_size", "800x600"))
            self.theme_var.set(ui_config.get("theme", "default"))
            self.progress_interval_var.set(ui_config.get("progress_update_interval", 100))
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    
    def save_settings(self) -> bool:
        """Save settings from form to config.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # ASR settings
            self.config.set("asr.model_name", self.asr_model_var.get())
            self.config.set("asr.device", self.asr_device_var.get())
            self.config.set("asr.batch_size", self.asr_batch_var.get())
            self.config.set("asr.chunk_length", self.asr_chunk_var.get())
            
            # Translation settings
            self.config.set("translation.ja_to_en_model", self.ja_en_model_var.get())
            self.config.set("translation.en_to_zh_model", self.en_zh_model_var.get())
            self.config.set("translation.device", self.trans_device_var.get())
            self.config.set("translation.batch_size", self.trans_batch_var.get())
            self.config.set("translation.max_length", self.trans_max_length_var.get())
            
            # Output settings
            self.config.set("output.default_format", self.output_format_var.get())
            self.config.set("output.include_confidence", self.include_confidence_var.get())
            self.config.set("output.output_directory", self.output_dir_var.get())
            self.config.set("output.temp_directory", self.temp_dir_var.get())
            
            # UI settings
            self.config.set("ui.window_size", self.window_size_var.get())
            self.config.set("ui.theme", self.theme_var.get())
            self.config.set("ui.progress_update_interval", self.progress_interval_var.get())
            
            # Save to file
            return self.config.save_config()
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
    
    def ok_clicked(self):
        """Handle OK button click."""
        if self.save_settings():
            self.result = True
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", "Failed to save settings.")
    
    def cancel_clicked(self):
        """Handle Cancel button click."""
        self.result = False
        self.dialog.destroy()
    
    def reset_clicked(self):
        """Handle Reset button click."""
        if messagebox.askyesno("Reset Settings", "Reset all settings to default values?"):
            # Reset config to defaults
            self.config.config = self.config.DEFAULT_CONFIG.copy()
            
            # Reload form
            self.load_settings()
    
    def show(self) -> bool:
        """Show the dialog and wait for result.
        
        Returns:
            True if OK was clicked, False if cancelled
        """
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self.cancel_clicked)
        
        # Show dialog
        self.dialog.wait_window()
        
        return self.result