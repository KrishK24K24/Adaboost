# Holiday Package Purchase Prediction using AdaBoost

A machine learning project that predicts whether customers will purchase a new Wellness Tourism Package using ensemble learning techniques, specifically focusing on AdaBoost classification.

## 📋 Project Overview

**Trips & Travel.Com** wants to expand its customer base by introducing a new Wellness Tourism Package. The company previously experienced an 18% conversion rate but incurred high marketing costs due to random customer outreach. This project leverages historical customer data to optimize marketing spend by predicting which customers are most likely to purchase the new package.

### Business Problem

- Current conversion rate: 18%
- Issue: High marketing costs from untargeted customer contact
- Goal: Use predictive modeling to identify high-potential customers and reduce marketing expenditure
- Target: Predict purchase likelihood for the new Wellness Tourism Package

## 🎯 Objectives

1. Build a predictive model to identify customers likely to purchase the package
2. Compare multiple classification algorithms to find the best performer
3. Optimize the selected model using hyperparameter tuning
4. Evaluate model performance using multiple metrics (accuracy, precision, recall, F1-score, ROC-AUC)

## 📊 Dataset

**Source**: [Kaggle - Holiday Package Purchase Prediction](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)

**Size**: 4,888 rows × 20 columns

### Features

| Feature | Description | Type |
|---------|-------------|------|
| CustomerID | Unique customer identifier | Numeric |
| ProdTaken | Target variable (1: Purchased, 0: Not purchased) | Binary |
| Age | Customer age | Numeric |
| TypeofContact | Contact method (Self Enquiry/Company Invited) | Categorical |
| CityTier | City classification tier | Categorical |
| DurationOfPitch | Sales pitch duration (minutes) | Numeric |
| Occupation | Customer occupation | Categorical |
| Gender | Customer gender | Categorical |
| NumberOfPersonVisiting | Number of people in travel group | Numeric |
| NumberOfFollowups | Follow-up count | Numeric |
| ProductPitched | Type of package pitched | Categorical |
| PreferredPropertyStar | Preferred hotel star rating | Numeric |
| MaritalStatus | Marital status | Categorical |
| NumberOfTrips | Historical trip count | Numeric |
| Passport | Passport ownership (1: Yes, 0: No) | Binary |
| PitchSatisfactionScore | Satisfaction with sales pitch | Numeric |
| OwnCar | Car ownership (1: Yes, 0: No) | Binary |
| NumberOfChildrenVisiting | Number of children in group | Numeric |
| Designation | Job designation | Categorical |
| MonthlyIncome | Monthly income | Numeric |

## 🛠️ Technical Stack

- **Python 3.x**
- **Libraries**:
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
    - Algorithms: AdaBoost, Random Forest, Gradient Boosting, Decision Tree, Logistic Regression
    - Preprocessing: StandardScaler, OneHotEncoder, ColumnTransformer
    - Model Selection: train_test_split, RandomizedSearchCV
    - Metrics: accuracy_score, classification_report, confusion_matrix, ROC-AUC

## 📈 Methodology

### 1. Data Cleaning & Preprocessing

**Handling Missing Values**:
- Age: Imputed with median
- TypeofContact: Imputed with mode
- DurationOfPitch: Imputed with median
- NumberOfFollowups: Imputed with mode (discrete feature)
- PreferredPropertyStar: Imputed with mode
- NumberOfTrips: Imputed with median
- NumberOfChildrenVisiting: Imputed with median
- MonthlyIncome: Imputed with median

**Data Quality Fixes**:
- Gender: Corrected 'Fe Male' → 'Female'
- MaritalStatus: Standardized 'Single' → 'Unmarried'

**Duplicate Handling**: Checked and removed duplicates

**Data Type Verification**: Ensured appropriate data types for all features

### 2. Feature Engineering

- **Feature Extraction**: Analyzed and extracted relevant features
- **Encoding**: Applied OneHotEncoder for categorical variables
- **Scaling**: Applied StandardScaler for numerical features
- **Pipeline**: Created ColumnTransformer for streamlined preprocessing

### 3. Model Development

**Train-Test Split**: 80-20 split with `random_state=42`

**Models Evaluated**:
1. Logistic Regression (baseline)
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. **AdaBoost Classifier** (primary focus)

### 4. Hyperparameter Tuning

**RandomizedSearchCV** applied to:
- Random Forest
- AdaBoost

**Optimized AdaBoost Parameters**:
- `n_estimators`: 80
- `algorithm`: 'SAMME'

**Optimized Random Forest Parameters**:
- `n_estimators`: 1000
- `min_samples_split`: 2
- `max_features`: 7
- `max_depth`: None

### 5. Model Evaluation

**Metrics**:
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve (AUC ≈ 0.6049 for AdaBoost)

## 🎓 Key Insights

1. **AdaBoost Performance**: The ensemble method effectively handles the class imbalance in the dataset
2. **Feature Importance**: Duration of pitch, number of follow-ups, and monthly income are significant predictors
3. **Model Comparison**: AdaBoost and Random Forest showed competitive performance after hyperparameter tuning
4. **Business Impact**: Targeted marketing based on predictions can significantly reduce customer acquisition costs

## 📂 Project Structure

```
AdaboostClassification/
│
├── AdaboostClassification.ipynb    # Main Jupyter notebook
├── Travel.csv                       # Dataset (not included in repo)
└── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Notebook

1. Clone the repository
```bash
git clone <repository-url>
cd AdaboostClassification
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction) and place it in the project directory as `Travel.csv`

3. Launch Jupyter Notebook
```bash
jupyter notebook AdaboostClassification.ipynb
```

4. Run all cells sequentially

## 📊 Results Summary

The optimized AdaBoost classifier achieved:
- **ROC-AUC Score**: ~0.60
- Successfully identified key customer segments for targeted marketing
- Provided actionable insights for reducing marketing costs

## 🔮 Future Enhancements

- [ ] Implement SMOTE or other resampling techniques for class imbalance
- [ ] Feature selection using recursive feature elimination (RFE)
- [ ] Deep learning approaches (Neural Networks)
- [ ] Real-time prediction API deployment
- [ ] A/B testing framework for production validation
- [ ] Advanced ensemble methods (XGBoost, LightGBM, CatBoost)

## 📝 License

This project is available for educational and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👨‍💻 Author

**Krish**

## 🙏 Acknowledgments

- Dataset: [Kaggle - Holiday Package Purchase Prediction](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)
- scikit-learn documentation and community
- Trips & Travel.Com (hypothetical business case)

---

**Note**: This project demonstrates the application of ensemble learning techniques for business intelligence and customer analytics. The focus on AdaBoost showcases how boosting algorithms can improve prediction accuracy by combining multiple weak learners into a strong classifier.
