"""Main application entry point for Sakura Subtitle Generator."""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.utils.logger import setup_logger, suppress_noisy_loggers
from src.ui.main_window import MainWindow


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sakura Subtitle Generator - Japanese ASR with Multi-language Translation"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in CLI mode (not implemented yet)"
    )
    
    parser.add_argument(
        "video_file",
        nargs="?",
        type=Path,
        help="Video file to process (for CLI mode)"
    )
    
    return parser.parse_args()


def setup_application(args):
    """Setup application configuration and logging."""
    # Load configuration
    config = Config(config_file=args.config)
    
    # Setup logging
    log_level = args.log_level or config.get("logging.level", "INFO")
    log_file = config.get("logging.file")
    log_format = config.get("logging.format")
    
    logger = setup_logger(
        name="sakura_subtitle",
        level=log_level,
        log_file=Path(log_file) if log_file else None,
        log_format=log_format
    )
    
    # Suppress noisy third-party loggers
    suppress_noisy_loggers()
    
    # Create necessary directories
    config.setup_directories()
    
    logger.info("🌸 Sakura Subtitle Generator starting up")
    logger.info(f"Configuration loaded from: {config.config_file}")
    
    return config, logger


def run_gui_mode(config):
    """Run application in GUI mode."""
    try:
        app = MainWindow(config)
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error running GUI: {e}")
        sys.exit(1)


def run_cli_mode(config, video_file):
    """Run application in CLI mode (future implementation)."""
    print("CLI mode is not implemented yet.")
    print("Please run without --no-gui flag to use the graphical interface.")
    sys.exit(1)


def main():
    """Main application entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup application
        config, logger = setup_application(args)
        
        # Run appropriate mode
        if args.no_gui:
            if not args.video_file:
                print("Error: Video file required for CLI mode")
                sys.exit(1)
            run_cli_mode(config, args.video_file)
        else:
            run_gui_mode(config)
            
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
