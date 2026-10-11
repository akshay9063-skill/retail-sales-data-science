# Project Report: Retail Sales Data Analysis

## 1. Problem statement
Retail businesses need to understand which products, categories, regions, and months contribute most to sales. This project analyses a sample dataset and demonstrates a baseline model for estimating order sales.

## 2. Objectives
- Prepare and validate tabular sales data.
- Summarise sales trends across time, category, and region.
- Build and evaluate a baseline regression model.
- Convert analysis into practical recommendations.

## 3. Methodology
1. Load the CSV using pandas.
2. Parse dates, remove incomplete rows and duplicate records.
3. Explore sales totals and averages.
4. Create monthly, category, and regional visualisations.
5. Train a Random Forest regression pipeline with a held-out test set.
6. Report MAE, RMSE, and R² metrics.

## 4. Initial findings
- Dataset size: 1,200 sample orders.
- Total sales: ₹4,614,193.72.
- Average order value: ₹3,845.16.
- Highest-sales category: Electronics.
- Highest-sales region: South.

## 5. Recommendations
- Use monthly patterns to plan inventory and promotional campaigns.
- Compare regional results before distributing marketing resources.
- Evaluate discounts together with sales and units sold.
- Re-train and validate the model on genuine, approved business data before real-world use.

## 6. Limitations
The provided data is synthetic and generated for this portfolio project. Results demonstrate the workflow, not the performance of a real retailer. Prediction metrics should be reviewed after running the script and should not be interpreted as proof of business impact.

## 7. Conclusion
The project demonstrates a complete introductory data science workflow—from CSV preparation to exploratory analysis, visualisation, and baseline machine learning—with reproducible code and documented outputs.
