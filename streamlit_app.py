# This file exists so Streamlit Cloud auto-detects the entry point.
# It simply re-exports app/app.py by setting the path and importing.
# Streamlit Cloud can also be pointed directly at app/app.py.
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "app.py")).read())
