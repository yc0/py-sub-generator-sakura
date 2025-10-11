"""Progress dialog for showing processing status."""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ProgressDialog:
    """Dialog for showing progress during processing."""
    
    def __init__(self, parent: tk.Tk, title: str = "Processing"):
        """Initialize progress dialog.
        
        Args:
            parent: Parent window
            title: Dialog title
        """
        self.parent = parent
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center on parent
        self.center_on_parent()
        
        # Create widgets
        self.create_widgets()
        
        # State
        self.cancelled = False
        self.cancel_callback: Optional[Callable] = None
    
    def center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        # Get parent position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate center position
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def create_widgets(self):
        """Create dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Arial", 10)
        )
        status_label.pack(pady=(0, 15))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=350
        )
        self.progress_bar.pack(pady=(0, 15))
        
        # Percentage label
        self.percentage_var = tk.StringVar(value="0%")
        percentage_label = ttk.Label(
            main_frame,
            textvariable=self.percentage_var,
            font=("Arial", 9),
            foreground="gray"
        )
        percentage_label.pack(pady=(0, 15))
        
        # Cancel button
        self.cancel_btn = ttk.Button(
            main_frame,
            text="Cancel",
            command=self.cancel
        )
        self.cancel_btn.pack()
        
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self.cancel)
    
    def update_progress(self, stage: str, progress: float):
        """Update progress display.
        
        Args:
            stage: Current processing stage
            progress: Progress value (0.0 to 1.0)
        """
        try:
            progress_percent = progress * 100
            
            self.status_var.set(stage)
            self.progress_var.set(progress_percent)
            self.percentage_var.set(f"{progress_percent:.1f}%")
            
            # Update display
            self.dialog.update()
            
        except Exception as e:
            logger.error(f"Error updating progress dialog: {e}")
    
    def set_cancel_callback(self, callback: Callable):
        """Set callback to call when cancel is pressed.
        
        Args:
            callback: Function to call on cancel
        """
        self.cancel_callback = callback
    
    def cancel(self):
        """Handle cancel button press."""
        if self.cancel_callback:
            self.cancel_callback()
        
        self.cancelled = True
        self.close()
    
    def close(self):
        """Close the dialog."""
        try:
            self.dialog.destroy()
        except:
            pass
    
    def is_cancelled(self) -> bool:
        """Check if dialog was cancelled.
        
        Returns:
            True if cancelled, False otherwise
        """
        return self.cancelled
    
    def show(self):
        """Show the dialog."""
        self.dialog.deiconify()
        self.dialog.lift()
        self.dialog.focus_force()
    
    def hide(self):
        """Hide the dialog."""
        self.dialog.withdraw()