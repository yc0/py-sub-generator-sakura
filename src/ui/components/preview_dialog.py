"""Preview dialog for displaying generated subtitles."""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Dict, List
import logging

from ...models.subtitle_data import SubtitleFile, SubtitleSegment

logger = logging.getLogger(__name__)


class PreviewDialog:
    """Dialog for previewing generated subtitles."""
    
    def __init__(self, parent: tk.Tk, subtitle_file: SubtitleFile):
        """Initialize preview dialog.
        
        Args:
            parent: Parent window
            subtitle_file: SubtitleFile to preview
        """
        self.parent = parent
        self.subtitle_file = subtitle_file
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Subtitle Preview")
        self.dialog.geometry("900x700")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        
        # Center on parent
        self.center_on_parent()
        
        # Create widgets
        self.create_widgets()
        
        # Load initial content
        self.load_preview()
    
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
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Language selection
        ttk.Label(controls_frame, text="Language:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.language_var = tk.StringVar()
        self.language_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.language_var,
            state="readonly",
            width=20
        )
        self.language_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.language_combo.bind("<<ComboboxSelected>>", self.on_language_changed)
        
        # Format selection
        ttk.Label(controls_frame, text="Format:").pack(side=tk.LEFT, padx=(10, 5))
        
        self.format_var = tk.StringVar(value="Formatted")
        format_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.format_var,
            values=["Formatted", "SRT", "Raw Text"],
            state="readonly",
            width=15
        )
        format_combo.pack(side=tk.LEFT, padx=(0, 10))
        format_combo.bind("<<ComboboxSelected>>", self.on_format_changed)
        
        # Refresh button
        ttk.Button(
            controls_frame,
            text="🔄 Refresh",
            command=self.load_preview
        ).pack(side=tk.RIGHT)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="")
        self.stats_label.pack()
        
        # Preview text area
        text_frame = ttk.LabelFrame(main_frame, text="Preview", padding="5")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Text widget with scrollbar
        self.preview_text = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            state=tk.DISABLED
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        # Close button
        ttk.Button(
            button_frame,
            text="Close",
            command=self.close
        ).pack(side=tk.RIGHT)
        
        # Export current view button
        ttk.Button(
            button_frame,
            text="💾 Export This View",
            command=self.export_current_view
        ).pack(side=tk.RIGHT, padx=(0, 10))
    
    def setup_language_options(self):
        """Setup available language options."""
        languages = ["Japanese (Original)"]
        
        # Add translations
        if self.subtitle_file.translations:
            lang_names = {
                'en': 'English',
                'zh': 'Traditional Chinese'
            }
            
            for lang_code in self.subtitle_file.translations.keys():
                lang_name = lang_names.get(lang_code, lang_code)
                languages.append(f"{lang_name} ({lang_code})")
        
        self.language_combo['values'] = languages
        
        # Set default selection
        if languages:
            self.language_var.set(languages[0])
    
    def load_preview(self):
        """Load and display subtitle preview."""
        try:
            # Setup language options
            self.setup_language_options()
            
            # Update statistics
            self.update_statistics()
            
            # Load content
            self.update_preview_content()
            
        except Exception as e:
            logger.error(f"Error loading preview: {e}")
            self.show_error(f"Error loading preview: {e}")
    
    def update_statistics(self):
        """Update statistics display."""
        try:
            total_segments = len(self.subtitle_file.segments)
            
            if self.subtitle_file.segments:
                total_duration = (
                    self.subtitle_file.segments[-1].end_time - 
                    self.subtitle_file.segments[0].start_time
                )
                avg_duration = sum(
                    seg.end_time - seg.start_time 
                    for seg in self.subtitle_file.segments
                ) / total_segments
                
                minutes = int(total_duration // 60)
                seconds = int(total_duration % 60)
                
                stats_text = (
                    f"Total Segments: {total_segments} | "
                    f"Duration: {minutes:02d}:{seconds:02d} | "
                    f"Avg Segment: {avg_duration:.1f}s"
                )
                
                # Add translation info
                if self.subtitle_file.translations:
                    trans_count = len(self.subtitle_file.translations)
                    stats_text += f" | Translations: {trans_count}"
            else:
                stats_text = "No subtitle data available"
            
            self.stats_label.config(text=stats_text)
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
            self.stats_label.config(text="Error loading statistics")
    
    def update_preview_content(self):
        """Update preview content based on current selections."""
        try:
            language_selection = self.language_var.get()
            format_selection = self.format_var.get()
            
            # Determine language
            if "Japanese" in language_selection:
                content = self.get_original_content(format_selection)
            else:
                # Extract language code from selection
                lang_code = None
                if "English" in language_selection:
                    lang_code = 'en'
                elif "Chinese" in language_selection:
                    lang_code = 'zh'
                
                if lang_code and lang_code in self.subtitle_file.translations:
                    content = self.get_translated_content(lang_code, format_selection)
                else:
                    content = "Translation not available for selected language."
            
            # Update text widget
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, content)
            self.preview_text.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Error updating preview content: {e}")
            self.show_error(f"Error updating content: {e}")
    
    def get_original_content(self, format_type: str) -> str:
        """Get original (Japanese) content in specified format."""
        segments = self.subtitle_file.segments
        
        if format_type == "SRT":
            return self.subtitle_file.export_srt()
        elif format_type == "Raw Text":
            return "\n".join(seg.text for seg in segments if seg.text.strip())
        else:  # Formatted
            return self.format_segments_display(segments)
    
    def get_translated_content(self, lang_code: str, format_type: str) -> str:
        """Get translated content in specified format."""
        translations = self.subtitle_file.translations.get(lang_code, [])
        
        if not translations:
            return f"No translations available for {lang_code}"
        
        if format_type == "SRT":
            return self.subtitle_file.export_srt(lang_code)
        elif format_type == "Raw Text":
            return "\n".join(
                trans.translated_text 
                for trans in translations 
                if trans.translated_text.strip()
            )
        else:  # Formatted
            # Create segments with translated text
            translated_segments = []
            for i, (seg, trans) in enumerate(zip(self.subtitle_file.segments, translations)):
                translated_seg = SubtitleSegment(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=trans.translated_text,
                    confidence=trans.confidence,
                    speaker_id=seg.speaker_id
                )
                translated_segments.append(translated_seg)
            
            return self.format_segments_display(translated_segments)
    
    def format_segments_display(self, segments: List[SubtitleSegment]) -> str:
        """Format segments for display."""
        lines = []
        
        for i, segment in enumerate(segments, 1):
            # Time formatting
            start_time = self.format_time(segment.start_time)
            end_time = self.format_time(segment.end_time)
            
            # Confidence info
            conf_info = ""
            if segment.confidence is not None:
                conf_info = f" (conf: {segment.confidence:.2f})"
            
            # Format segment
            lines.append(f"[{i:3d}] {start_time} → {end_time}{conf_info}")
            lines.append(f"      {segment.text}")
            lines.append("")  # Empty line
        
        return "\n".join(lines)
    
    def format_time(self, seconds: float) -> str:
        """Format time in MM:SS.mmm format."""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    def on_language_changed(self, event=None):
        """Handle language selection change."""
        self.update_preview_content()
    
    def on_format_changed(self, event=None):
        """Handle format selection change."""
        self.update_preview_content()
    
    def export_current_view(self):
        """Export current view to file."""
        try:
            from tkinter import filedialog
            
            # Determine file extension based on format
            format_selection = self.format_var.get()
            if format_selection == "SRT":
                default_ext = ".srt"
                filetypes = [("SRT files", "*.srt"), ("All files", "*.*")]
            else:
                default_ext = ".txt"
                filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
            
            # Get filename
            language_selection = self.language_var.get()
            lang_suffix = ""
            if "English" in language_selection:
                lang_suffix = "_en"
            elif "Chinese" in language_selection:
                lang_suffix = "_zh"
            elif "Japanese" in language_selection:
                lang_suffix = "_ja"
            
            default_name = f"subtitles_preview{lang_suffix}{default_ext}"
            
            file_path = filedialog.asksaveasfilename(
                title="Export Preview",
                defaultextension=default_ext,
                initialname=default_name,
                filetypes=filetypes
            )
            
            if file_path:
                # Get current content
                content = self.preview_text.get(1.0, tk.END).rstrip()
                
                # Write to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                tk.messagebox.showinfo("Export Complete", f"Preview exported to:\n{file_path}")
                
        except Exception as e:
            logger.error(f"Error exporting preview: {e}")
            tk.messagebox.showerror("Export Error", f"Failed to export preview: {e}")
    
    def show_error(self, message: str):
        """Show error message in preview area."""
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(1.0, f"Error: {message}")
        self.preview_text.config(state=tk.DISABLED)
    
    def close(self):
        """Close the dialog."""
        self.dialog.destroy()
    
    def show(self):
        """Show the dialog."""
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self.close)
        
        # Show dialog
        self.dialog.deiconify()
        self.dialog.lift()
        self.dialog.focus_force()