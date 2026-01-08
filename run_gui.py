#!/usr/bin/env python3
"""
DetectAI Training Manager - PyQt GUI Application
Launch with: python3 run_gui.py
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from app.gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DetectAI Training Manager")
    
    # Set default font
    font = QFont("SF Pro", 13)
    app.setFont(font)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
