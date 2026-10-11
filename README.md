# Retail Sales Data Science Project

## Project overview
This project demonstrates an end-to-end data science workflow using a retail sales dataset: data loading, cleaning, exploratory data analysis, visualisation, a simple prediction model, and business recommendations.

**Important:** `data/retail_sales.csv` is a reproducible *sample/synthetic dataset created for learning*. It is not confidential customer data and should not be described as a real company's sales records.

## Project goals
- Analyse sales by month, product category, and region.
- Identify useful trends and high-performing categories.
- Train a baseline machine-learning model to estimate sales value.
- Present results through charts and a dashboard screenshot.

## Tools
- Python
- pandas and NumPy
- Matplotlib
- scikit-learn

## Repository structure
```text
retail-sales-data-science/
├── data/
│   └── retail_sales.csv
├── outputs/
│   ├── category_sales.png
│   ├── dashboard_screenshot.png
│   ├── monthly_sales.png
│   ├── region_sales.png
│   └── summary.json
├── src/
│   └── analyze_sales.py
├── requirements.txt
└── README.md
```

## Run locally
1. Install Python 3.10 or newer.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the analysis:
   ```bash
   python src/analyze_sales.py
   ```

The script prints the main findings, evaluates a baseline Random Forest model using a held-out test set, and regenerates the charts in `outputs/`.

## Current sample-data snapshot
- Orders analysed: **1,200**
- Total sales: **₹4,614,193.72**
- Average order value: **₹3,845.16**
- Highest-sales category: **Electronics**
- Highest-sales region: **South**

Exact results are based on the included synthetic sample data and may change if the dataset is replaced.

## Business recommendations
1. Track monthly sales to plan stock and promotions.
2. Compare category and regional performance before allocating marketing budget.
3. Examine discounts alongside units sold and sales value.
4. Treat the prediction model as a baseline, not a production forecasting system.

## Limitations
- The included dataset is synthetic and is intended for education and demonstration.
- This baseline model estimates order sales from available columns; it does not prove causation.
- A real deployment would require validated business data, stronger evaluation, and ongoing monitoring.

## Suggested GitHub repository details
**Repository name:** `retail-sales-data-science`

**Description:** `End-to-end retail sales data analysis project using Python, pandas, visualizations, and a baseline machine-learning model.`

**Commit message:** `Add retail sales analysis project with dataset and visualizations`
