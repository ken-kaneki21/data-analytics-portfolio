\# Telemetry Anomaly Detection (Time-Series ML) — NASA CMAPSS FD001



\## Problem

Detect early abnormal behavior in multivariate telemetry time-series data. Using NASA CMAPSS turbofan engine sensor streams, we model normal operating behavior during early life and assign anomaly scores over time to flag abnormal conditions prior to failure.



\## Data

Dataset: NASA CMAPSS (FD001). Each engine produces sequential telemetry per operating cycle (3 operating settings + 21 sensors).  

We build a canonical table with `engine\_id`, `cycle`, operating settings, sensors, and train-only `cycles\_to\_failure`.



\## Method (Baseline)

We start with simple statistical and classical ML baselines (e.g., z-score drift, Isolation Forest) to produce cycle-level anomaly scores and identify top contributing sensors.



\## Results

TBD (added after baseline models + evaluation).



\## Dashboard

TBD (Power BI pages: Fleet Overview + Engine Drilldown style).



\## How to Run

1\. Place CMAPSS files in `data/raw/CMAPSSData/`:

&nbsp;  - `train\_FD001.txt`, `test\_FD001.txt`, `RUL\_FD001.txt`

2\. Create venv and install:

&nbsp;  - `pip install -r requirements.txt`

3\. Run ingestion:

&nbsp;  - `python src/01\_ingest.py`



