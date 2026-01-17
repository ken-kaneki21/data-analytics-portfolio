EV Demand Forecasting – India (2W) | ML \& Time Series

Project Overview



This project focuses on predictive modeling of Electric Vehicle (EV) demand in India, with an emphasis on 2-Wheeler (2W) adoption forecasting using machine learning and time-series feature engineering.



This work is intentionally distinct from my earlier EV Market Study – India project, which focuses purely on descriptive BI and historical trend analysis.

Here, the goal is forecast reliability, model validation, and decision-oriented insights.



🎯 Business Objective



Forecast monthly EV demand to support capacity planning, infrastructure readiness, and policy analysis



Evaluate whether EV demand is predictable and learnable across vehicle categories



Clearly identify where forecasting is not reliable and explain why



📊 Data Source



Clean Mobility Shift EV Dashboard



Data aggregated from the Indian Government’s VAHAN (RTO) database



Monthly EV registrations (2014–2024)



Categories: 2-Wheelers and 4-Wheelers



All data sources are publicly available and cited for transparency.



🧠 Modeling Approach

1\. Feature Engineering



Lag features: lag\_1, lag\_3, lag\_6, lag\_12



Rolling statistics: 3-month \& 6-month rolling mean and standard deviation



Monthly aggregation to stabilize noise



2\. Models Evaluated



Prophet (trend + seasonality)



Random Forest (lag-based ML model)



3\. Model Selection

Category	Model	Performance

2-Wheelers	Random Forest	27.4% MAPE

4-Wheelers	Any model	❌ Unreliable (>80% sMAPE)



Prophet was rejected due to poor validation stability.

Random Forest demonstrated better generalization for 2W demand.



⚠️ Why 4-Wheeler Forecasting Was Rejected



4-Wheeler EV sales in India exhibit:



Low absolute volumes



Irregular spikes



Weak and inconsistent seasonality



Back testing showed high forecast error (>80% sMAPE), making predictions unreliable.

Therefore:



4W analysis is restricted to descriptive trends and YoY growth diagnostics



Forecasting is intentionally excluded and documented



This modeling judgment is a key outcome of the project.



📈 Outputs



Monthly demand forecasts for 2-Wheelers



Forecast CSVs for downstream reporting



Power BI dashboards:



Actual vs ML Forecast (2W)



Historical trends \& YoY growth (4W)



🛠 Tools \& Tech



Python (Pandas, NumPy, Scikit-learn, Prophet)



Power BI (DAX, time-series visuals)



Machine Learning (Random Forest regression)



Git \& GitHub



Key Takeaways



2-Wheeler EV demand in India is learnable and forecastable



Not all business problems should be forecasted — knowing when not to predict is critical



Clear separation between descriptive BI and predictive analytics improves decision quality

