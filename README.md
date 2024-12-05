# Automatic License Plate Regconition

## 1. Create New Folder and Navigate to New Folder
Open CMD and Run as Administrator
    
    cd C:\Path\To\Your\Project
## 2. Create Virtual Environment and install package

    
    git clone https://github.com/KYUNSSSS/ALPR
    cd ALPR
    python -m venv vir_env
    vir_env\Scripts\activate
    
    pip install -r requirements.txt
    

## 3. RUN APP (For Streamlit)
    streamlit run paddleOCRYOLO.py
## 3.1 RUN APP (For Flask)
    python app.py

- Copy the http link generated to open through browser
- CTRL+C to stop the app 
## 4. Deactivate virtual environment (When Finish)
    deactivate 

## 5. To Activate Again
     vir_env\Scripts\activate



