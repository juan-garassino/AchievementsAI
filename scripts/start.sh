#!/bin/bash

# Start the Ollama service in the background
ollama serve &

# Start Streamlit
streamlit run AchievementsAI.py --server.port=8501 --server.address=0.0.0.0
