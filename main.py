import subprocess
from pathlib import Path

def launch_app():
    app_path = Path(__file__).parent / "app.py"
    print(f"🚀 Đang khởi động Giao diện Web PPE Detection từ: {app_path}")
    
    venv_streamlit = Path(__file__).parent / ".venv" / "Scripts" / "streamlit.exe"
    executable = str(venv_streamlit) if venv_streamlit.exists() else "streamlit"
    
    cmd = [executable, "run", str(app_path)]
    subprocess.run(cmd)

if __name__ == "__main__":
    launch_app()
