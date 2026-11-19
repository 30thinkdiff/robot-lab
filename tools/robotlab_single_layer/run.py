"""Portable entry without colliding with ROS's installed `scripts` package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robotlab_single_layer.__main__ import main

if __name__ == '__main__':
    main()
