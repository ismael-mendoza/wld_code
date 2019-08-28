import sys 
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root1 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, root1)

import preamble
import weights
import maps
import money