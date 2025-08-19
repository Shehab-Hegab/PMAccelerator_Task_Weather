## Global Weather & Climate Insights Dashboard

**Project Mission**: In alignment with The PM Accelerator's mission — "To empower the next generation of product leaders with the practical skills, mentorship, and community needed to accelerate their careers."

### Project Overview
This project delivers an end-to-end data science workflow to analyze the Kaggle "Global Weather Repository" dataset, build forecasting models, and present insights in an interactive Streamlit dashboard. It spans the full lifecycle: data ingestion, cleaning, exploratory data analysis (EDA), geospatial analytics, modeling, evaluation, and deployment in a user-friendly web app.

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

![Streamlit Dashboard](Images/Screenshot%202025-08-19%20191204.png)

![Global Map Example](Images/download.png)

### Media Gallery (from `Images/`)
All attached images are listed here for quick reference.

![download (1).png](Images/download%20(1).png)
![download (2).png](Images/download%20(2).png)
![download (3).png](Images/download%20(3).png)
![download (4).png](Images/download%20(4).png)
![download (5).png](Images/download%20(5).png)
![download (6).png](Images/download%20(6).png)
![download (7).png](Images/download%20(7).png)
![download (8).png](Images/download%20(8).png)
![download (9).png](Images/download%20(9).png)
![download (10).png](Images/download%20(10).png)
![download (11).png](Images/download%20(11).png)
![download (12).png](Images/download%20(12).png)
![download (13).png](Images/download%20(13).png)
![download (14).png](Images/download%20(14).png)
![download (15).png](Images/download%20(15).png)
![download.png](Images/download.png)
![download.png1.png](Images/download.png1.png)
![Screenshot 2025-08-19 171359.png](Images/Screenshot%202025-08-19%20171359.png)
![Screenshot 2025-08-19 171422.png](Images/Screenshot%202025-08-19%20171422.png)
![Screenshot 2025-08-19 171511.png](Images/Screenshot%202025-08-19%20171511.png)
![Screenshot 2025-08-19 171550.png](Images/Screenshot%202025-08-19%20171550.png)
![Screenshot 2025-08-19 191204.png](Images/Screenshot%202025-08-19%20191204.png)

Video (if supported by your README viewer): `Images/Recording 2025-08-19 171624.mp4`

### Acknowledgments
- Natural Earth for country boundaries.
- Kaggle community for the "Global Weather Repository" dataset.
- The PM Accelerator for inspiring the applied product mindset behind this work.


