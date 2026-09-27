* Evaluates model performance across three target variables: sleep efficiency, minutes asleep, and restfulness
* Uses 10-fold cross-validation grouped by participant ID to prevent data leakage across subjects 
* Filters out invalid records where minutes asleep is zero or phone usage duration is missing   
* Baseline Model: Evaluates a mean dummy regressor to establish threshold RMSE performance   
* Random Forest Model: Evaluates non-linear regressors across single, paired, and fully combined feature groups
* Configures Random Forest hyperparameters with 100 trees, a maximum depth of 10, and a minimum leaf size of 5
* Imputes missing predictor values using feature medians while tracking missingness via binary indicator flags   