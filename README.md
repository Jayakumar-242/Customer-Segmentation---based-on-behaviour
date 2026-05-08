# 🎯 Customer Segmentation using Machine Learning

Divide customers into meaningful segments using K-Means clustering on demographic and behavioral data.

## Project Structure
```
Customer Segmentation/
├── data/customers.csv
├── notebooks/analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── clustering.py
│   ├── visualization.py
│   └── utils.py
├── dashboard/app.py
├── outputs/
│   ├── charts/
│   └── segmented_customers.csv
├── main.py
├── generate_data.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generate_data.py

# 3. Run full pipeline (charts + CSV output)
python main.py

# 4. Launch interactive dashboard
streamlit run dashboard/app.py
```

## Features
- K-Means clustering with elbow + silhouette analysis
- PCA 2D & interactive 3D visualizations
- Professional Streamlit dashboard with filters & download
- Segment profiling and labeling

## Tech Stack
Python · Pandas · Scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit
