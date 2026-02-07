import os
import subprocess
import sys

print("Installing Python dependencies...")

# Install from requirements.txt
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print("Checking PyAudio...")
try:
    import pyaudio
    print("PyAudio installed ✅")
except ImportError:
    print("PyAudio not installed! Please follow OS-specific instructions:")
    print("Windows: download .whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
    print("Linux: sudo apt install portaudio19-dev python3-pyaudio")
    print("macOS: brew install portaudio && pip install pyaudio")

print("Setup complete! Run the assistant with:")
print("python assistant.py")
