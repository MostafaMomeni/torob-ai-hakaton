# Ensure repository root is on sys.path when running pytest so top-level modules import correctly.
import os, sys
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
