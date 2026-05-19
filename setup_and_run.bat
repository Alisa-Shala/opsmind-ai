@echo off
echo ============================================
echo  OpsMind AI - Setup and Launch
echo ============================================
echo.

echo [1/5] Installing dependencies...
pip install -r requirements.txt -q

echo [2/5] Generating invoice dataset...
py src/generate_dataset.py

echo [3/5] Cleaning and processing data...
py src/data_cleaning.py

echo [4/5] Loading data to database...
py src/load_to_database.py

echo [5/5] Training ML model...
py src/train_models.py

echo.
echo ============================================
echo  Setup complete! Launching dashboard...
echo ============================================
echo.
py -m streamlit run src/app.py
