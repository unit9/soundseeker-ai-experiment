import os

CV_KEY = os.environ.get('CV_KEY') or EnvironmentError("Env var CV_KEY (Google Vision Api Key) is missing")

Y_ALL_PATH = os.environ.get('Y_ALL_PATH', 'data/ai/Y_all.npy')
MUSIC_CSV_PATH = os.environ.get('MUSIC_CSV_PATH', 'data/ai_music/music.csv')
MODEL_DIR = os.environ.get('MODEL_DIR', 'data/ai')