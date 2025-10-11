"""Sakura Subtitle Generator - Japanese ASR with Multi-language Translation."""

__version__ = "1.0.0"
__author__ = "Sakura Team"
__description__ = "🌸 A powerful tool for generating Japanese subtitles with multi-language translation support"

from .utils.config import Config
from .utils.logger import setup_logger
from .subtitle.subtitle_generator import SubtitleGenerator

__all__ = [
    'Config',
    'setup_logger', 
    'SubtitleGenerator'
]