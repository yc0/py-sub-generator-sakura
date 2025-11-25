"""Settings dialog for configuring application parameters."""

import logging
import tkinter as tk
from tkinter import messagebox, ttk

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

        # Calculate responsive dialog size based on screen resolution
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()

        # Use 70% of screen height or minimum 700px, whichever is larger
        dialog_height = max(700, int(screen_height * 0.7))
        # Outer window 850px to accommodate scrollbar + content without overlap
        dialog_width = 850

        self.dialog.geometry(f"{dialog_width}x{dialog_height}")
        self.dialog.minsize(850, 650)  # Ensure minimum usable size
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
        """Center dialog on parent window, ensuring it fits on screen."""
        self.dialog.update_idletasks()

        # Get screen dimensions
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()

        # Get dialog dimensions
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()

        # Try to center on parent if parent is visible
        try:
            parent_x = self.parent.winfo_rootx()
            parent_y = self.parent.winfo_rooty()
            parent_width = self.parent.winfo_width()
            parent_height = self.parent.winfo_height()

            x = parent_x + (parent_width // 2) - (dialog_width // 2)
            y = parent_y + (parent_height // 2) - (dialog_height // 2)
        except:
            # Fallback to screen center
            x = (screen_width // 2) - (dialog_width // 2)
            y = (screen_height // 2) - (dialog_height // 2)

        # Ensure dialog stays within screen bounds
        x = max(0, min(x, screen_width - dialog_width))
        y = max(0, min(y, screen_height - dialog_height))

        self.dialog.geometry(f"+{x}+{y}")

    def _get_available_sakura_models(self):
        """Get list of available SakuraLLM models from config."""
        sakura_config = self.config.get_sakura_config()
        available_models_config = sakura_config.get("available_models", {})
        
        model_display_list = []
        
        # Add models from available_models config
        for model_key, model_info in available_models_config.items():
            description = model_info.get("description", "")
            vram = model_info.get("vram_required", "Unknown")
            display_name = f"{model_key} ({vram}) - {description}"
            model_display_list.append(display_name)
        
        # Fallback if no models configured
        if not model_display_list:
            model_display_list = [
                "sakura-7b-v1.0 (8GB) - Balanced 7B parameter model (IQ4_XS quantized) - Recommended",
                "sakura-14b-v1.0 (16GB) - High-quality 14B parameter model (IQ4_XS quantized) - Best Quality",
            ]
        
        return model_display_list

    def _get_display_name_for_model(self, model_name):
        """Get display name for a given model name."""
        sakura_config = self.config.get_sakura_config()
        available_models_config = sakura_config.get("available_models", {})
        
        # Find matching model by model_name
        for model_key, model_info in available_models_config.items():
            if model_info.get("model_name") == model_name:
                description = model_info.get("description", "")
                vram = model_info.get("vram_required", "Unknown")
                return f"{model_key} ({vram}) - {description}"
        
        # Fallback mapping for compatibility
        model_mapping = {
            "SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF": "sakura-7b-v1.0 (8GB) - Balanced 7B parameter model (IQ4_XS quantized) - Recommended",
            "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF": "sakura-14b-v1.0 (16GB) - High-quality 14B parameter model (IQ4_XS quantized) - Best Quality",
        }
        
        return model_mapping.get(model_name, "sakura-7b-v1.0 (8GB) - Balanced 7B parameter model (IQ4_XS quantized) - Recommended")

    def _get_model_key_from_display(self, display_text):
        """Extract model key from display text."""
        # Extract the model key from display format: "sakura-7b-v1.0 (8GB) - Description"
        if display_text and " (" in display_text:
            model_key = display_text.split(" (")[0]
            return model_key
        
        # Fallback to default
        return "sakura-7b-v1.0"

    def create_widgets(self):
        """Create dialog widgets."""
        # Main container frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Create scrollable frame for the notebook
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create notebook for tabs
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_to_mousewheel)
        canvas.bind("<Leave>", _unbind_from_mousewheel)

        # Store references for button frame
        self.main_frame = main_frame
        self.scrollable_frame = scrollable_frame

        # ASR Settings Tab
        self.create_asr_tab(notebook)

        # Translation Settings Tab
        self.create_translation_tab(notebook)

        # Output Settings Tab
        self.create_output_tab(notebook)

        # UI Settings Tab
        self.create_ui_tab(notebook)

        # Button frame (outside scrollable area)
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))

        # Buttons
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(
            side=tk.RIGHT, padx=(5, 0)
        )

        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(
            side=tk.RIGHT
        )

        ttk.Button(
            button_frame, text="Reset to Default", command=self.reset_clicked
        ).pack(side=tk.LEFT)

    def create_asr_tab(self, notebook):
        """Create ASR settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="ASR")

        # Model selection
        ttk.Label(frame, text="Whisper Model:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
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
                "kotoba-tech/kotoba-whisper-v2.1",
                "kotoba-tech/kotoba-whisper-v2.2",
                "openai/whisper-large-v3",
            ],
            state="readonly",
        )
        model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # Device selection
        ttk.Label(frame, text="Device:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.asr_device_var = tk.StringVar()
        device_combo = ttk.Combobox(
            frame,
            textvariable=self.asr_device_var,
            values=["auto", "cpu", "cuda", "mps"],
            state="readonly",
        )
        device_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # Batch size
        ttk.Label(frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.asr_batch_var = tk.IntVar()
        batch_spin = tk.Spinbox(
            frame, from_=1, to=8, textvariable=self.asr_batch_var, width=10
        )
        batch_spin.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Chunk length
        ttk.Label(frame, text="Chunk Length (seconds):").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.asr_chunk_var = tk.IntVar()
        chunk_spin = tk.Spinbox(
            frame, from_=5, to=60, textvariable=self.asr_chunk_var, width=10
        )
        chunk_spin.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Overlap duration
        ttk.Label(frame, text="Overlap (seconds):").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.asr_overlap_var = tk.DoubleVar()
        overlap_spin = tk.Spinbox(
            frame,
            from_=0.0,
            to=2.0,
            increment=0.1,
            format="%.1f",
            textvariable=self.asr_overlap_var,
            width=10,
        )
        overlap_spin.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Native Whisper info
        info_frame = ttk.Frame(frame)
        info_frame.grid(
            row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(15, 5)
        )

        info_label = ttk.Label(
            info_frame,
            text="ℹ️  Using Whisper's native generate() method - no experimental warnings, no token limits!",
            foreground="green",
            font=("TkDefaultFont", 8),
        )
        info_label.grid(row=0, column=0, sticky=tk.W)

        # Configure column weights
        frame.columnconfigure(1, weight=1)

    def create_translation_tab(self, notebook):
        """Create unified translation settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Translation")

        # Translation method selection
        method_frame = ttk.LabelFrame(frame, text="Translation Method", padding="10")
        method_frame.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        self.translation_method_var = tk.StringVar(value="huggingface")

        # HuggingFace Multi-Stage option
        hf_radio = ttk.Radiobutton(
            method_frame,
            text="🔄 Multi-Stage Translation (Japanese → English → Chinese)",
            variable=self.translation_method_var,
            value="huggingface",
            command=self.on_translation_method_change,
        )
        hf_radio.grid(row=0, column=0, sticky=tk.W, pady=5)

        hf_desc = ttk.Label(
            method_frame,
            text="Uses separate models for ja→en and en→zh translation. Supports multiple language pairs.",
            font=("TkDefaultFont", 9),
            foreground="gray",
        )
        hf_desc.grid(row=1, column=0, sticky=tk.W, padx=(20, 0), pady=(0, 10))

        # SakuraLLM Direct option
        sakura_radio = ttk.Radiobutton(
            method_frame,
            text="🌸 SakuraLLM Direct Translation (Japanese → Chinese)",
            variable=self.translation_method_var,
            value="sakura",
            command=self.on_translation_method_change,
        )
        sakura_radio.grid(row=2, column=0, sticky=tk.W, pady=5)

        sakura_desc = ttk.Label(
            method_frame,
            text="High-quality specialized model for Japanese→Chinese translation. Optimized for anime/manga content.",
            font=("TkDefaultFont", 9),
            foreground="gray",
        )
        sakura_desc.grid(row=3, column=0, sticky=tk.W, padx=(20, 0), pady=(0, 5))

        method_frame.columnconfigure(0, weight=1)

        # HuggingFace Settings Frame
        self.hf_frame = ttk.LabelFrame(
            frame, text="🔄 Multi-Stage Translation Settings", padding="10"
        )
        self.hf_frame.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        # Japanese to English model
        ttk.Label(self.hf_frame, text="Japanese→English Model:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.ja_en_model_var = tk.StringVar()
        ja_en_entry = ttk.Entry(
            self.hf_frame, textvariable=self.ja_en_model_var, width=50
        )
        ja_en_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # English to Chinese model
        ttk.Label(self.hf_frame, text="English→Chinese Model:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.en_zh_model_var = tk.StringVar()
        en_zh_entry = ttk.Entry(
            self.hf_frame, textvariable=self.en_zh_model_var, width=50
        )
        en_zh_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # HuggingFace device
        ttk.Label(self.hf_frame, text="Device:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.trans_device_var = tk.StringVar()
        trans_device_combo = ttk.Combobox(
            self.hf_frame,
            textvariable=self.trans_device_var,
            values=["auto", "cpu", "cuda", "mps"],
            state="readonly",
        )
        trans_device_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # HuggingFace batch size
        ttk.Label(self.hf_frame, text="Batch Size:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.trans_batch_var = tk.IntVar()
        trans_batch_spin = tk.Spinbox(
            self.hf_frame, from_=1, to=32, textvariable=self.trans_batch_var, width=10
        )
        trans_batch_spin.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Max length
        ttk.Label(self.hf_frame, text="Max Sequence Length:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.trans_max_length_var = tk.IntVar()
        max_length_spin = tk.Spinbox(
            self.hf_frame,
            from_=128,
            to=1024,
            increment=64,
            textvariable=self.trans_max_length_var,
            width=10,
        )
        max_length_spin.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        self.hf_frame.columnconfigure(1, weight=1)

        # SakuraLLM Settings Frame
        self.sakura_frame = ttk.LabelFrame(
            frame, text="🌸 SakuraLLM Translation Settings", padding="10"
        )
        self.sakura_frame.grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        # Model selection
        ttk.Label(self.sakura_frame, text="Model Size:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.sakura_model_var = tk.StringVar()
        # Dynamically get available models from config
        available_models = self._get_available_sakura_models()
        model_combo = ttk.Combobox(
            self.sakura_frame,
            textvariable=self.sakura_model_var,
            values=available_models,
            state="readonly",
            width=60,
        )
        model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # SakuraLLM device selection
        ttk.Label(self.sakura_frame, text="Device:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.sakura_device_var = tk.StringVar()
        sakura_device_combo = ttk.Combobox(
            self.sakura_frame,
            textvariable=self.sakura_device_var,
            values=["auto", "cuda", "mps", "cpu"],
            state="readonly",
        )
        sakura_device_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Temperature
        ttk.Label(self.sakura_frame, text="Temperature (0.0-1.0):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.sakura_temp_var = tk.DoubleVar()
        temp_scale = ttk.Scale(
            self.sakura_frame,
            from_=0.0,
            to=1.0,
            variable=self.sakura_temp_var,
            orient=tk.HORIZONTAL,
            length=200,
        )
        temp_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # Temperature value label
        self.temp_value_label = ttk.Label(self.sakura_frame, text="0.1")
        self.temp_value_label.grid(row=2, column=2, padx=(10, 0), pady=5)

        # Update temperature display
        def update_temp_display(*args):
            self.temp_value_label.config(text=f"{self.sakura_temp_var.get():.1f}")

        self.sakura_temp_var.trace("w", update_temp_display)

        # Max new tokens
        ttk.Label(self.sakura_frame, text="Max New Tokens:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.sakura_max_tokens_var = tk.IntVar()
        tokens_spin = tk.Spinbox(
            self.sakura_frame,
            from_=128,
            to=2048,
            increment=128,
            textvariable=self.sakura_max_tokens_var,
            width=10,
        )
        tokens_spin.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Advanced options frame
        adv_frame = ttk.Frame(self.sakura_frame)
        adv_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # Force GPU
        self.sakura_force_gpu_var = tk.BooleanVar()
        gpu_check = ttk.Checkbutton(
            adv_frame,
            text="Force GPU acceleration (recommended)",
            variable=self.sakura_force_gpu_var,
        )
        gpu_check.grid(row=0, column=0, sticky=tk.W, pady=2)

        # Use chat template
        self.sakura_chat_template_var = tk.BooleanVar()
        template_check = ttk.Checkbutton(
            adv_frame,
            text="Use chat template (better translation quality)",
            variable=self.sakura_chat_template_var,
        )
        template_check.grid(row=1, column=0, sticky=tk.W, pady=2)

        self.sakura_frame.columnconfigure(1, weight=1)

        frame.columnconfigure(0, weight=1)

    def on_translation_method_change(self):
        """Handle translation method change."""
        method = self.translation_method_var.get()

        if method == "huggingface":
            # Enable HuggingFace frame, disable SakuraLLM frame
            self._set_frame_state(self.hf_frame, tk.NORMAL)
            self._set_frame_state(self.sakura_frame, tk.DISABLED)
        else:  # sakura
            # Enable SakuraLLM frame, disable HuggingFace frame
            self._set_frame_state(self.hf_frame, tk.DISABLED)
            self._set_frame_state(self.sakura_frame, tk.NORMAL)

    def _set_frame_state(self, frame, state):
        """Recursively set the state of all widgets in a frame."""
        for child in frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                # Some widgets don't support state configuration
                if hasattr(child, "winfo_children"):
                    self._set_frame_state(child, state)

    def create_output_tab(self, notebook):
        """Create output settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Output")

        # Default format
        ttk.Label(frame, text="Default Format:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.output_format_var = tk.StringVar()
        format_combo = ttk.Combobox(
            frame,
            textvariable=self.output_format_var,
            values=["srt", "vtt", "ass"],
            state="readonly",
        )
        format_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Include confidence
        self.include_confidence_var = tk.BooleanVar()
        confidence_check = ttk.Checkbutton(
            frame,
            text="Include confidence scores",
            variable=self.include_confidence_var,
        )
        confidence_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Output directory
        ttk.Label(frame, text="Output Directory:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.output_dir_var = tk.StringVar()
        output_dir_entry = ttk.Entry(frame, textvariable=self.output_dir_var, width=40)
        output_dir_entry.grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5
        )

        # Temp directory
        ttk.Label(frame, text="Temp Directory:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.temp_dir_var = tk.StringVar()
        temp_dir_entry = ttk.Entry(frame, textvariable=self.temp_dir_var, width=40)
        temp_dir_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # Dual language output section
        separator = ttk.Separator(frame, orient="horizontal")
        separator.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)

        # Generate both languages option
        self.generate_both_var = tk.BooleanVar()
        dual_lang_check = ttk.Checkbutton(
            frame,
            text="Generate both original (Japanese) and translated subtitle files",
            variable=self.generate_both_var,
            command=self._on_dual_language_toggle,
        )
        dual_lang_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Info label for dual language feature
        info_label = ttk.Label(
            frame,
            text="💡 When enabled, creates separate files for Japanese and translated subtitles\n"
                 "   Example: video_ja.srt (Japanese) and video_zh.srt (Chinese)",
            foreground="blue",
            font=("TkDefaultFont", 8),
        )
        info_label.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # Multi-language combination option
        self.combine_languages_var = tk.BooleanVar()
        combine_lang_check = ttk.Checkbutton(
            frame,
            text="Combine all languages into a single SRT file",
            variable=self.combine_languages_var,
        )
        combine_lang_check.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Info label for multi-language combination feature
        combine_info_label = ttk.Label(
            frame,
            text="💡 When enabled, creates one file with all languages combined\n"
                 "   Example: video_en_jp_zh.srt (English, Japanese, Chinese)",
            foreground="green",
            font=("TkDefaultFont", 8),
        )
        combine_info_label.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # Language suffix customization (initially hidden)
        self.suffix_frame = ttk.LabelFrame(frame, text="File Naming Options", padding="5")

        # Original language suffix
        ttk.Label(self.suffix_frame, text="Japanese file suffix:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.original_suffix_var = tk.StringVar()
        original_suffix_entry = ttk.Entry(
            self.suffix_frame, textvariable=self.original_suffix_var, width=10
        )
        original_suffix_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)

        # Translated language suffix
        ttk.Label(self.suffix_frame, text="Chinese file suffix:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.translated_suffix_var = tk.StringVar()
        translated_suffix_entry = ttk.Entry(
            self.suffix_frame, textvariable=self.translated_suffix_var, width=10
        )
        translated_suffix_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)

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
            state="readonly",
        )
        size_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Theme
        ttk.Label(frame, text="Theme:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.theme_var = tk.StringVar()
        theme_combo = ttk.Combobox(
            frame,
            textvariable=self.theme_var,
            values=["default", "clam", "alt", "classic"],
            state="readonly",
        )
        theme_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Progress update interval
        ttk.Label(frame, text="Progress Update Interval (ms):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.progress_interval_var = tk.IntVar()
        interval_spin = tk.Spinbox(
            frame,
            from_=50,
            to=500,
            increment=50,
            textvariable=self.progress_interval_var,
            width=10,
        )
        interval_spin.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        frame.columnconfigure(1, weight=1)

    def _on_dual_language_toggle(self):
        """Handle dual language checkbox toggle."""
        if self.generate_both_var.get():
            # Show suffix configuration options
            self.suffix_frame.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        else:
            # Hide suffix configuration options
            self.suffix_frame.grid_remove()

    def load_settings(self):
        """Load current settings into form."""
        try:
            # ASR settings
            asr_config = self.config.get_asr_config()
            self.asr_model_var.set(
                asr_config.get("model_name", "openai/whisper-large-v3")
            )
            self.asr_device_var.set(asr_config.get("device", "auto"))
            self.asr_batch_var.set(asr_config.get("batch_size", 1))
            self.asr_chunk_var.set(asr_config.get("chunk_length", 30))
            self.asr_overlap_var.set(asr_config.get("overlap", 0.5))

            # Translation settings
            trans_config = self.config.get_translation_config()
            self.ja_en_model_var.set(
                trans_config.get("ja_to_en_model", "Helsinki-NLP/opus-mt-ja-en")
            )
            self.en_zh_model_var.set(
                trans_config.get("en_to_zh_model", "Helsinki-NLP/opus-mt-en-zh")
            )
            self.trans_device_var.set(trans_config.get("device", "auto"))
            self.trans_batch_var.set(trans_config.get("batch_size", 8))
            self.trans_max_length_var.set(trans_config.get("max_length", 512))

            # Translation method and SakuraLLM settings
            sakura_config = self.config.get_sakura_config()
            is_sakura_enabled = sakura_config.get("enabled", False)

            # Set translation method based on SakuraLLM enabled status
            self.translation_method_var.set(
                "sakura" if is_sakura_enabled else "huggingface"
            )

            # Map model name to display text dynamically
            current_model = sakura_config.get("model_name", "")
            display_model = self._get_display_name_for_model(current_model)
            self.sakura_model_var.set(display_model)

            self.sakura_device_var.set(sakura_config.get("device", "auto"))
            self.sakura_temp_var.set(sakura_config.get("temperature", 0.1))
            self.sakura_max_tokens_var.set(sakura_config.get("max_new_tokens", 512))
            self.sakura_force_gpu_var.set(sakura_config.get("force_gpu", True))
            self.sakura_chat_template_var.set(
                sakura_config.get("use_chat_template", True)
            )

            # Update frame states based on selected method
            self.on_translation_method_change()

            # Output settings
            output_config = self.config.get_output_config()
            self.output_format_var.set(output_config.get("default_format", "srt"))
            self.include_confidence_var.set(
                output_config.get("include_confidence", True)
            )
            self.output_dir_var.set(output_config.get("output_directory", "outputs"))
            self.temp_dir_var.set(output_config.get("temp_directory", "temp"))

            # Dual language settings
            self.generate_both_var.set(output_config.get("generate_both_languages", True))
            self.original_suffix_var.set(output_config.get("original_language_suffix", "_ja"))
            self.translated_suffix_var.set(output_config.get("translated_language_suffix", "_zh"))

            # Multi-language combination setting
            self.combine_languages_var.set(output_config.get("combine_languages", False))

            # Trigger toggle to show/hide suffix options
            self._on_dual_language_toggle()

            # UI settings
            ui_config = self.config.get_ui_config()
            self.window_size_var.set(ui_config.get("window_size", "800x600"))
            self.theme_var.set(ui_config.get("theme", "default"))
            self.progress_interval_var.set(
                ui_config.get("progress_update_interval", 100)
            )

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
            self.config.set("asr.overlap", self.asr_overlap_var.get())

            # Translation settings
            self.config.set("translation.ja_to_en_model", self.ja_en_model_var.get())
            self.config.set("translation.en_to_zh_model", self.en_zh_model_var.get())
            self.config.set("translation.device", self.trans_device_var.get())
            self.config.set("translation.batch_size", self.trans_batch_var.get())
            self.config.set("translation.max_length", self.trans_max_length_var.get())

            # Translation method and SakuraLLM settings
            is_sakura_method = self.translation_method_var.get() == "sakura"
            self.config.set("sakura.enabled", is_sakura_method)

            # Map display text back to model name dynamically
            selected_display = self.sakura_model_var.get()
            model_key = self._get_model_key_from_display(selected_display)

            # Set the model using the config's helper method
            if is_sakura_method:
                self.config.set_sakura_model(model_key)

            self.config.set("sakura.device", self.sakura_device_var.get())
            self.config.set("sakura.temperature", self.sakura_temp_var.get())
            self.config.set("sakura.max_new_tokens", self.sakura_max_tokens_var.get())
            self.config.set("sakura.force_gpu", self.sakura_force_gpu_var.get())
            self.config.set(
                "sakura.use_chat_template", self.sakura_chat_template_var.get()
            )

            # Output settings
            self.config.set("output.default_format", self.output_format_var.get())
            self.config.set(
                "output.include_confidence", self.include_confidence_var.get()
            )
            self.config.set("output.output_directory", self.output_dir_var.get())
            self.config.set("output.temp_directory", self.temp_dir_var.get())

            # Dual language settings
            self.config.set("output.generate_both_languages", self.generate_both_var.get())
            self.config.set("output.original_language_suffix", self.original_suffix_var.get())
            self.config.set("output.translated_language_suffix", self.translated_suffix_var.get())

            # Multi-language combination setting
            self.config.set("output.combine_languages", self.combine_languages_var.get())

            # UI settings
            self.config.set("ui.window_size", self.window_size_var.get())
            self.config.set("ui.theme", self.theme_var.get())
            self.config.set(
                "ui.progress_update_interval", self.progress_interval_var.get()
            )

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
        if messagebox.askyesno(
            "Reset Settings", "Reset all settings to default values?"
        ):
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
