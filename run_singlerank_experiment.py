import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.main_singlerank import main

if __name__ == "__main__":
    main()