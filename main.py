from src.train import train_model
import subprocess

if __name__ == "__main__":
    file_path = "data\IMDB Dataset.csv"
    model, tokenizer = train_model(file_path)

    api_process = subprocess.Popen(["python", 'api/app.py'])
    streamlit_process = subprocess.Popen(["streamlit", 'run', 'frontend/streamlitApp.py'])
    try:
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        api_process.terminate()
        streamlit_process.terminate()