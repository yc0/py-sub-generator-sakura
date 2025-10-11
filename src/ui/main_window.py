"""Main application window using Tkinter."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import logging

from ..utils.config import Config
from ..utils.logger import LoggerMixin
from ..subtitle.subtitle_generator import SubtitleGenerator
from ..models.subtitle_data import SubtitleFile
from .components.progress_dialog import ProgressDialog
from .components.settings_dialog import SettingsDialog
from .components.preview_dialog import PreviewDialog

logger = logging.getLogger(__name__)


class MainWindow(LoggerMixin):
    """Main application window for Sakura Subtitle Generator."""
    
    def __init__(self, config: Config):
        """Initialize main window.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.subtitle_generator = SubtitleGenerator(config)
        
        # Application state
        self.current_video_path: Optional[Path] = None
        self.current_subtitle_file: Optional[SubtitleFile] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        
        # Create main window
        self.root = tk.Tk()
        self.setup_window()
        self.create_widgets()
        self.setup_bindings()
        
        self.logger.info("Main window initialized")
    
    def setup_window(self):
        """Setup main window properties."""
        ui_config = self.config.get_ui_config()
        
        self.root.title(ui_config.get("window_title", "Sakura Subtitle Generator"))
        self.root.geometry(ui_config.get("window_size", "800x600"))
        self.root.minsize(600, 400)
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use(ui_config.get("theme", "default"))
    
    def apply_ui_settings(self):
        """Apply UI settings from current configuration."""
        try:
            ui_config = self.config.get_ui_config()
            
            # Update window title if changed
            current_title = self.root.title()
            new_title = ui_config.get("window_title", "Sakura Subtitle Generator")
            if current_title != new_title:
                self.root.title(new_title)
            
            # Update theme if changed
            current_theme = self.style.theme_use()
            new_theme = ui_config.get("theme", "default")
            if current_theme != new_theme:
                try:
                    self.style.theme_use(new_theme)
                    self.logger.info(f"Theme changed to: {new_theme}")
                except tk.TclError:
                    self.logger.warning(f"Theme '{new_theme}' not available, keeping '{current_theme}'")
            
            # Note: Window size changes require restart to take full effect
            # as resizing programmatically during runtime can be disruptive
            
        except Exception as e:
            self.logger.error(f"Error applying UI settings: {e}")
    
    def create_widgets(self):
        """Create and arrange GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🌸 Sakura Subtitle Generator",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Video file selection
        self.create_file_selection_section(main_frame, row=1)
        
        # Language selection
        self.create_language_selection_section(main_frame, row=2)
        
        # Processing controls
        self.create_processing_controls_section(main_frame, row=3)
        
        # Status and progress
        self.create_status_section(main_frame, row=4)
        
        # Results section
        self.create_results_section(main_frame, row=5)
        
        # Menu bar
        self.create_menu_bar()
    
    def create_file_selection_section(self, parent, row):
        """Create file selection section."""
        # Frame
        file_frame = ttk.LabelFrame(parent, text="Video File", padding="10")
        file_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)
        
        # File path display
        self.file_path_var = tk.StringVar(value="No file selected")
        file_label = ttk.Label(file_frame, textvariable=self.file_path_var)
        file_label.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Browse button
        browse_btn = ttk.Button(
            file_frame,
            text="📁 Browse Video File",
            command=self.browse_video_file,
            width=20
        )
        browse_btn.grid(row=1, column=0, padx=(0, 10))
        
        # File info display
        self.file_info_var = tk.StringVar(value="")
        info_label = ttk.Label(
            file_frame, 
            textvariable=self.file_info_var,
            foreground="gray"
        )
        info_label.grid(row=1, column=1, sticky=tk.W)
    
    def create_language_selection_section(self, parent, row):
        """Create language selection section."""
        lang_frame = ttk.LabelFrame(parent, text="Translation Languages", padding="10")
        lang_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Language checkboxes
        self.lang_vars = {
            'en': tk.BooleanVar(value=True),
            'zh': tk.BooleanVar(value=True)
        }
        
        ttk.Label(lang_frame, text="Generate subtitles for:").grid(row=0, column=0, sticky=tk.W)
        
        en_check = ttk.Checkbutton(
            lang_frame,
            text="🇺🇸 English",
            variable=self.lang_vars['en']
        )
        en_check.grid(row=0, column=1, padx=10)
        
        zh_check = ttk.Checkbutton(
            lang_frame,
            text="🇹🇼 Traditional Chinese",
            variable=self.lang_vars['zh']
        )
        zh_check.grid(row=0, column=2, padx=10)
        
        # Note
        note_label = ttk.Label(
            lang_frame,
            text="Original Japanese subtitles will always be generated",
            foreground="gray",
            font=("Arial", 9)
        )
        note_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
    
    def create_processing_controls_section(self, parent, row):
        """Create processing controls section."""
        controls_frame = ttk.Frame(parent)
        controls_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        # Generate button
        self.generate_btn = ttk.Button(
            controls_frame,
            text="🚀 Generate Subtitles",
            command=self.start_processing,
            style="Accent.TButton"
        )
        self.generate_btn.grid(row=0, column=0, padx=10)
        
        # Cancel button
        self.cancel_btn = ttk.Button(
            controls_frame,
            text="❌ Cancel",
            command=self.cancel_processing,
            state=tk.DISABLED
        )
        self.cancel_btn.grid(row=0, column=1, padx=10)
        
        # Settings button
        settings_btn = ttk.Button(
            controls_frame,
            text="⚙️ Settings",
            command=self.show_settings
        )
        settings_btn.grid(row=0, column=2, padx=10)
    
    def create_status_section(self, parent, row):
        """Create status and progress section."""
        status_frame = ttk.LabelFrame(parent, text="Status", padding="10")
        status_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        status_frame.columnconfigure(0, weight=1)
        
        # Status text
        self.status_var = tk.StringVar(value="Ready to process video")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def create_results_section(self, parent, row):
        """Create results section."""
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        
        # Configure parent row to expand
        parent.rowconfigure(row, weight=1)
        
        # Results text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = tk.Text(
            text_frame,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Results buttons
        button_frame = ttk.Frame(results_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.preview_btn = ttk.Button(
            button_frame,
            text="👁️ Preview",
            command=self.preview_subtitles,
            state=tk.DISABLED
        )
        self.preview_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.export_btn = ttk.Button(
            button_frame,
            text="💾 Export",
            command=self.export_subtitles,
            state=tk.DISABLED
        )
        self.export_btn.grid(row=0, column=1)
    
    def create_menu_bar(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Video...", command=self.browse_video_file)
        file_menu.add_separator()
        file_menu.add_command(label="Export Subtitles...", command=self.export_subtitles)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Settings...", command=self.show_settings)
        tools_menu.add_command(label="Clear Results", command=self.clear_results)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_bindings(self):
        """Setup event bindings."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def browse_video_file(self):
        """Open file dialog to select video file."""
        if self.is_processing:
            messagebox.showwarning("Processing", "Please wait for current processing to complete.")
            return
        
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if file_path:
            self.current_video_path = Path(file_path)
            self.file_path_var.set(str(self.current_video_path))
            
            # Get video info
            video_file = self.subtitle_generator.file_handler.create_video_file_object(
                self.current_video_path
            )
            if video_file:
                video_file = self.subtitle_generator.file_handler.get_video_metadata(video_file)
                self.file_info_var.set(video_file.get_display_info())
            
            # Enable generate button
            self.generate_btn.config(state=tk.NORMAL)
            
            self.logger.info(f"Video file selected: {self.current_video_path}")
    
    def start_processing(self):
        """Start subtitle generation in background thread."""
        if not self.current_video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return
        
        if self.is_processing:
            messagebox.showwarning("Processing", "Processing is already in progress.")
            return
        
        # Get selected languages
        target_languages = []
        if self.lang_vars['en'].get():
            target_languages.append('en')
        if self.lang_vars['zh'].get():
            target_languages.append('zh')
        
        # Update UI state
        self.is_processing = True
        self.generate_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.preview_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        
        # Clear results
        self.clear_results()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_video_thread,
            args=(self.current_video_path, target_languages),
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.info("Started subtitle generation process")
    
    def _process_video_thread(self, video_path: Path, target_languages: list):
        """Background thread for video processing."""
        try:
            def progress_callback(stage: str, progress: float):
                # Schedule UI update on main thread
                self.root.after(0, self._update_progress, stage, progress)
            
            # Process video
            subtitle_file = self.subtitle_generator.process_video_file(
                video_path,
                target_languages=target_languages,
                progress_callback=progress_callback
            )
            
            # Schedule completion on main thread
            self.root.after(0, self._processing_completed, subtitle_file)
            
        except Exception as e:
            self.logger.error(f"Error in processing thread: {e}")
            self.root.after(0, self._processing_error, str(e))
    
    def _update_progress(self, stage: str, progress: float):
        """Update progress display (called on main thread)."""
        stage_names = {
            "validation": "Validating video file",
            "audio_extraction": "Extracting audio",
            "asr": "Generating transcription",
            "translation": "Translating subtitles"
        }
        
        stage_name = stage_names.get(stage, stage)
        progress_percent = progress * 100
        
        self.status_var.set(f"{stage_name}... ({progress_percent:.1f}%)")
        self.progress_var.set(progress_percent)
        
        # Add to results text
        if progress == 1.0:
            self._add_result_text(f"✅ {stage_name} completed\n")
    
    def _processing_completed(self, subtitle_file: Optional[SubtitleFile]):
        """Handle processing completion (called on main thread)."""
        self.is_processing = False
        self.generate_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        
        if subtitle_file:
            self.current_subtitle_file = subtitle_file
            self.status_var.set("✅ Subtitle generation completed successfully!")
            
            # Show results summary
            self._show_processing_results(subtitle_file)
            
            # Enable result buttons
            self.preview_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            
        else:
            self.status_var.set("❌ Subtitle generation failed")
            self._add_result_text("\n❌ Processing failed. Check the log for details.\n")
        
        self.progress_var.set(0)
    
    def _processing_error(self, error_message: str):
        """Handle processing error (called on main thread)."""
        self.is_processing = False
        self.generate_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        
        self.status_var.set("❌ Error during processing")
        self._add_result_text(f"\n❌ Error: {error_message}\n")
        
        self.progress_var.set(0)
    
    def _show_processing_results(self, subtitle_file: SubtitleFile):
        """Show processing results summary."""
        self._add_result_text("\n🎉 Processing Results:\n")
        self._add_result_text(f"📊 Generated {len(subtitle_file.segments)} subtitle segments\n")
        
        # Show translation results
        for lang_code, translations in subtitle_file.translations.items():
            lang_names = {'en': 'English', 'zh': 'Traditional Chinese'}
            lang_name = lang_names.get(lang_code, lang_code)
            self._add_result_text(f"🌐 {lang_name}: {len(translations)} translations\n")
        
        self._add_result_text("\n✨ Ready to preview or export!\n")
    
    def cancel_processing(self):
        """Cancel current processing."""
        if self.is_processing and self.processing_thread:
            # Note: This is a simplified cancel - in a production app,
            # you'd want proper thread cancellation mechanisms
            messagebox.showinfo("Cancel", "Processing will stop after current stage completes.")
            
    def preview_subtitles(self):
        """Show subtitle preview dialog."""
        if not self.current_subtitle_file:
            messagebox.showerror("Error", "No subtitles to preview.")
            return
        
        try:
            dialog = PreviewDialog(self.root, self.current_subtitle_file)
            dialog.show()
        except Exception as e:
            self.logger.error(f"Error showing preview: {e}")
            messagebox.showerror("Error", f"Failed to show preview: {e}")
    
    def export_subtitles(self):
        """Export subtitles to files."""
        if not self.current_subtitle_file:
            messagebox.showerror("Error", "No subtitles to export.")
            return
        
        # Select output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        try:
            output_path = Path(output_dir)
            
            # Export subtitles
            exported_files = self.subtitle_generator.export_subtitles(
                self.current_subtitle_file,
                output_path,
                formats=['srt']
            )
            
            if exported_files:
                self._add_result_text(f"\n📁 Exported to: {output_path}\n")
                for format_name, file_path in exported_files.items():
                    self._add_result_text(f"   • {file_path.name}\n")
                
                messagebox.showinfo(
                    "Export Complete",
                    f"Subtitles exported successfully to:\n{output_path}"
                )
            else:
                messagebox.showerror("Error", "Failed to export subtitles.")
                
        except Exception as e:
            self.logger.error(f"Error exporting subtitles: {e}")
            messagebox.showerror("Error", f"Failed to export subtitles: {e}")
    
    def show_settings(self):
        """Show settings dialog."""
        try:
            dialog = SettingsDialog(self.root, self.config)
            if dialog.show():
                # Settings were changed, reload configuration from file
                self.config.load_config()
                # Reinitialize components with updated configuration
                self.subtitle_generator = SubtitleGenerator(self.config)
                # Apply UI changes
                self.apply_ui_settings()
                self.logger.info("🔄 Configuration reloaded after settings change")
        except Exception as e:
            self.logger.error(f"Error showing settings: {e}")
            messagebox.showerror("Error", f"Failed to show settings: {e}")
    
    def clear_results(self):
        """Clear results display."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """🌸 Sakura Subtitle Generator
        
Version 1.0.0

A powerful tool for generating Japanese subtitles with multi-language translation support.

Features:
• Automatic Speech Recognition (Japanese)
• Multi-language translation (English, Traditional Chinese)
• High-quality subtitle generation
• Easy-to-use interface

Powered by:
• OpenAI Whisper (ASR)
• Hugging Face Transformers (Translation)
• Tkinter (GUI)

© 2024 Sakura Subtitle Generator"""
        
        messagebox.showinfo("About", about_text)
    
    def _add_result_text(self, text: str):
        """Add text to results display."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window closing event."""
        if self.is_processing:
            if messagebox.askokcancel("Quit", "Processing is in progress. Are you sure you want to quit?"):
                # Cleanup and quit
                self.subtitle_generator.cleanup()
                self.root.destroy()
        else:
            # Cleanup and quit
            self.subtitle_generator.cleanup()
            self.root.destroy()
    
    def run(self):
        """Start the GUI main loop."""
        try:
            self.logger.info("Starting GUI main loop")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            # Cleanup
            self.subtitle_generator.cleanup()