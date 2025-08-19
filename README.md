## Global Weather & Climate Insights Dashboard

**Project Mission**: In alignment with The PM Accelerator's mission — "To empower the next generation of product leaders with the practical skills, mentorship, and community needed to accelerate their careers."

### Project Overview
This project delivers an end-to-end data science workflow to analyze the Kaggle "Global Weather Repository" dataset, build forecasting models, and present insights in an interactive Streamlit dashboard. It spans the full lifecycle: data ingestion, cleaning, exploratory data analysis (EDA), geospatial analytics, modeling, evaluation, and deployment in a user-friendly web app.

https://github.com/user-attachments/assets/e808d5cc-90b9-43e5-ac1a-ff5df9c38f59

### Key Features
- **Interactive Global Maps**: Choropleth maps for country-level metrics, including average temperature, PM2.5 air quality (log scale), and a composite Climate Risk Index.
- **City-Specific Analysis**: An interactive sidebar to select a city and explore dedicated tabs for forecast, air quality correlation, and seasonal climate patterns.
- **Temperature Forecasting (POC)**: A proof-of-concept next-day temperature model (XGBoost) surfaced in the dashboard. The notebook includes additional experimentation.
- **Advanced Analytics**: Correlation analyses between air quality and temperature, plus seasonal pattern exploration across cities and time.

### Tech Stack
- **Core**: Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly
- **Geospatial**: GeoPandas (with shapefile support)
- **Modeling**: Scikit-learn, XGBoost, LightGBM, Statsmodels, TensorFlow/Keras, SHAP
- **App**: Streamlit

### Project Structure
- `PMAccelerator_Task_.ipynb`: Primary Jupyter Notebook with data prep, EDA, modeling experiments, and visuals.
- `app.py`: Streamlit application for the interactive dashboard (maps, city analytics, and forecast section).
- `forecasting_model.pkl`: Serialized forecasting model (loaded by the app).
- `GlobalWeatherRepository.csv`: Source data file used by the app.
- `ne_110m_admin_0_countries.*`: Natural Earth shapefile components for world map geometry.
- `dashboard_geodata.csv`, `london_sample_data.csv`, and other `.csv`/`.pkl`: Supporting datasets and artifacts.
- `Images/`: Screenshots and assets; a gallery is included below.

### Installation & Usage
1) Clone the repository
```bash
git clone <your-repo-url>.git
cd PMAccelerator_Task
```

2) Create a virtual environment (recommended) and install dependencies
```bash
# PowerShell (Windows)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Run the Streamlit app
```bash
streamlit run app.py
```

Notes:
- Place `GlobalWeatherRepository.csv` and the Natural Earth shapefiles (`ne_110m_admin_0_countries.*`) in the project root alongside `app.py`.
- The app loads `forecasting_model.pkl` from the project root.

### Key Insights & Visuals
- **Modeling**: In our proof-of-concept, gradient-boosted trees (XGBoost) provided the strongest next-day temperature baseline among evaluated approaches on the sample city data. See `PMAccelerator_Task_.ipynb` for training details and metrics.
- **Climate Risk Index**: A composite metric combining normalized temperature, PM2.5, and precipitation highlights hotspot regions on the choropleth map.

Featured visuals:


### Media Gallery (from `Images/`)
All attached images are listed here for quick reference.

<img width="1088" height="937" alt="Image" src="https://github.com/user-attachments/assets/d91cc033-59a4-4814-af5a-60ace196d9b7" />
<img width="790" height="580" alt="Image" src="https://github.com/user-attachments/assets/d2a67f44-1361-48d6-90a2-30c4f36c7ed3" />
<img width="765" height="579" alt="Image" src="https://github.com/user-attachments/assets/7ad8ed2c-706e-4b70-a191-1c666842efc3" />
<img width="1325" height="547" alt="Image" src="https://github.com/user-attachments/assets/d012f1f7-9dd5-4e49-b770-c15e792cab9f" />
<img width="776" height="684" alt="Image" src="https://github.com/user-attachments/assets/948efdca-5182-4fc5-aaab-423982b21694" />
https://github.com/user-attachments/assets/fb9f8140-3044-4ab3-b6fc-3f8d9561617e
<img width="1212" height="447" alt="Image" src="https://github.com/user-attachments/assets/7444017b-8595-4a8b-a552-3d6d761d5dac" />
<img width="1102" height="411" alt="Image" src="https://github.com/user-attachments/assets/8c6d9b99-c353-432b-953b-89c578144714" />
<img width="1211" height="452" alt="Image" src="https://github.com/user-attachments/assets/1104e44c-611f-4181-a0fe-b3465a3906ed" />
<img width="1227" height="778" alt="Image" src="https://github.com/user-attachments/assets/b03c557d-f5cc-44c7-b0cc-7951f9091be4" />
<img width="1657" height="418" alt="Image" src="https://github.com/user-attachments/assets/59aa7c52-7fdb-43f3-838c-84bce4efa31b" />
<img width="1022" height="553" alt="Image" src="https://github.com/user-attachments/assets/b57894ef-d656-404a-b423-3408121b8281" />
<img width="1000" height="553" alt="Image" src="https://github.com/user-attachments/assets/bb962a34-a723-43a3-b61b-1a53dfbdde1b" />
<img width="1621" height="711" alt="Image" src="https://github.com/user-attachments/assets/709c0c35-8c49-41c0-9df7-965e69edaa04" />
<img width="1472" height="630" alt="Image" src="https://github.com/user-attachments/assets/75353d83-266c-4449-a2d8-135cf742e397" />
<img width="1472" height="630" alt="Image" src="https://github.com/user-attachments/assets/69dadc88-68f6-47b5-b61b-2404ac5e7098" />
<img width="1589" height="790" alt="Image" src="https://github.com/user-attachments/assets/49b36fc6-b62f-48ef-a5fe-bb79270019c8" />
<img width="1043" height="706" alt="Image" src="https://github.com/user-attachments/assets/53c8334a-e022-4b09-8fd8-43e029b408a7" />
<img width="1163" height="706" alt="Image" src="https://github.com/user-attachments/assets/7b434e92-f297-4de8-9755-6d00a0a15662" />
<img width="1617" height="706" alt="Image" src="https://github.com/user-attachments/assets/cdda0209-2671-4488-8ec8-cbaa7726e49d" />
<img width="790" height="604" alt="Image" src="https://github.com/user-attachments/assets/06db9156-53df-4ced-8630-6aaf9e96172f" />
<img width="765" height="598" alt="Image" src="https://github.com/user-attachments/assets/d5ff80f9-f495-40e6-889d-dadda206cf76" />
<img width="996" height="573" alt="Image" src="https://github.com/user-attachments/assets/51160238-ef78-4f83-a834-9ce0fb3951d1" />

Video (if supported by your README viewer): `Images/Recording 2025-08-19 171624.mp4`

### Acknowledgments
- Natural Earth for country boundaries.
- Kaggle community for the "Global Weather Repository" dataset.
- The PM Accelerator for inspiring the applied product mindset behind this work.




