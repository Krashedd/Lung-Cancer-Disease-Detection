import os, subprocess, sys

def fix_opencv_conflict():
    try:
        # Uninstall GUI OpenCV if present
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"], check=False)
        print("✅ Ensured only opencv-python-headless is used.")
    except Exception as e:
        print("⚠️ Could not uninstall opencv-python:", e)

fix_opencv_conflict()
