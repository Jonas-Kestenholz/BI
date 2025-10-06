# Sprint 4 — Dashboard, Interpretation & Deployment

## Objectives
- Integrate all previous BI work (Sprints 1–3) into an interactive **Streamlit web app**.  
- Communicate results and insights visually through KPIs, EDA charts, and ML diagrams.  
- Demonstrate inference capability — predicting **spend** and **profit** directly from the app.  
- Reflect on model performance, business relevance, and next-step improvements.

---

## Methods
- **Framework**: Streamlit (Python, open-source BI app).  
- **Structure**:
  - *Overview*: project metrics and dataset info.  
  - *Data Preview*: quick dataset inspection (shape, describe).  
  - *Descriptive Diagrams*: EDA visuals grouped by category (Distributions, Relationships, Categories).  
  - *ML Diagrams*: model performance plots (regression, clustering, classification).  
  - *Predict Tabs*: interactive prediction for both Shopping and Superstore models.  
- **Features**:
  - Auto-grouped visuals (duplicates removed).  
  - Captions added under each figure for quick interpretation.  
  - User-friendly manual input forms for model inference.  
  - KPIs summarizing dataset size, total sales, and profit.  

---

## Model Justification
- **Regression (Profit Prediction)**:  
  - Started with *Linear Regression*, but *Random Forest Regressor* improved R² from 0.43 → 0.65 and reduced RMSE/MAE.  
  - Chosen for its ability to capture nonlinear effects (e.g., strong negative influence of discount).  
- **Clustering (Segmentation)**:  
  - *KMeans* + StandardScaler + Elbow + Silhouette → optimal K = 5.  
  - Used features like `Sales_Log`, `Profit_YJ`, `Discount`, and `ShippingCost_Log`.  
  - Segments interpreted as premium, average, and discount-heavy customer types.  
- **Classification (Profit/Loss)**:  
  - *Gaussian Naive Bayes* → Accuracy ≈ 91%, F1 ≈ 0.88.  
  - Simple, interpretable, and performs well with continuous/log-scaled data.

---

## Key Insights
- **Discount** is the strongest negative driver of profit (−0.59 corr).  
- **Technology** and **Copiers** categories deliver the highest returns.  
- Segmentation reveals 5 customer groups — one high-loss, one high-profit, and three average.  
- The models can guide pricing or discount limits in real time.  

---

## Deliverables
- **Streamlit App** → `/BI/app.py`  
- **Models** → `/BI/machine_learning/joblib/`  
- **EDA Figures** → `/BI/figures/`  
- **ML Figures** → `/BI/machine_learning/ml_results_diagrams/`  
- **Docs** → Sprint 1–4 Markdown files (this document).  

---

## Conclusion
Sprint 4 connects analytics to business value.  
The BI app combines data exploration, model results, and live prediction in one interface.  
This makes insights actionable — managers can now test *“what-if”* scenarios and see immediate model feedback.
