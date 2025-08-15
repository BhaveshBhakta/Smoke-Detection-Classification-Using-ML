## Smoke Detection Classification

### Project Overview

This project aims to classify the **presence of smoke or a fire alarm** based on sensor data from an IoT device. By analyzing various environmental parameters such as temperature, humidity, gas concentrations (TVOC, eCO2), and particle matter levels (PM1.0, PM2.5), the goal is to develop a machine learning model that can accurately detect fire alarms. This is crucial for building smart fire safety systems.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset)
  * **Size**: 62630 entries, 16 columns
  * **Key Features**:
      * `Temperature`, `Humidity`, `TVOC`, `eCO2`, `Raw H2`, `Raw Ethanol`, `Pressure`, `PM1.0`, `PM2.5`, `NC0.5`, `NC1.0`, `NC2.5`, `CNT` (sensor count).
  * **Approach**:
      * **Data Cleaning**: Dropped the 'Unnamed: 0' and highly correlated columns (`PM2.5`, `NC0.5`, `NC1.0`, `NC2.5`). No missing values or duplicates were found.
      * **Exploratory Data Analysis**: Histograms, boxplots, and a heatmap were used for visualization to understand data distributions and correlations.
      * **Handling Class Imbalance**: The dataset is imbalanced (44757 fires vs 17873 no fires). `SMOTETomek` was applied to the training data to balance the classes.
      * **Binary Classification**: The target variable `Fire Alarm` indicates the presence (`1`) or absence (`0`) of a fire alarm.
      * **Models Used**:
          * Logistic Regression, Ridge Classifier, SVC, Random Forest, XGBoost, AdaBoost, Gradient Boosting, Bagging, Decision Tree.
  * **Best Accuracy**:
      * **100%** with Random Forest, Gradient Boosting, and Bagging classifiers.
      * Other ensemble models also performed exceptionally well, with accuracies close to 100%. The very high accuracies suggest that the sensor data in this dataset provides extremely strong discriminative power for detecting a fire alarm.

-----

### Purpose and Applications

  * Enable **real-time smoke and fire detection** in smart homes and industrial environments.
  * Provide an automated and intelligent system for fire safety that is more reliable than traditional detectors.
  * Support data-driven decision-making in emergency services by providing early warnings.
  * Serve as a foundation for developing robust IoT-based safety systems.

-----

### Installation

Clone the repository:

```bash
git clone https://github.com/BhaveshBhakta/Smoke-Detection-Classification-Using-ML.git
cd Smoke-Detection-Classification-Using-ML
```

Install the necessary libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * Investigating the very high accuracy for potential data leakage, which is a significant concern for such a predictive task.
  * Performing comprehensive hyperparameter tuning and cross-validation for all models to ensure robustness.
  * Exploring more advanced feature engineering techniques, especially from the time-series nature of the data if available.
  * Adding explainability (e.g., SHAP or LIME) to understand which sensor readings are the most critical for smoke detection.
