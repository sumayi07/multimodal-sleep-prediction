# Multimodal Sleep Prediction

**Predicting sleep quality using exercise, mood, and phone data extracted from wearables and EMAs.** 

This project attempts to determine what factor or combination of factors of exercise, mood, and phone usage affects sleep quality the most.

## Data Source
* Data was extracted from the **TILES-2018 study**.

## Repository Structure
* `data/` – Scripts used to process and clean data from the TILES-2018 dataset.
* `models/` – Random Forest (RF) models used to evaluate cleaned data and evaluate their ability to predict sleep quality.
* `results/` - Compares the performance of the RF models to the baseline models using root mean square error

## Technology Stack
* **Language:** Python
* **Core Models:** Random Forest (RF)

## Citation

* Max Sun, Erica Wu. (2025). *Daily Sleep Prediction Using Wearable Sensors and Self-Reported Behavioral Metrics: A Multimodal Machine Learning Approach*. Presented at the Biomedical Engineering Society (BMES) Annual Meeting.
* Max Sun, Erica Wu. (2025). *Daily Sleep Prediction Using Wearable Sensors and Self-Reported Behavioral Metrics: A Multimodal Machine Learning Approach*. Presented at the IEEE Baltimore Technical & Professional Development Colloquium.
