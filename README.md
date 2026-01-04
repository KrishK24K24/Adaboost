# AdaBoost Machine Learning Projects

This repository contains two comprehensive machine learning projects demonstrating the application of AdaBoost (Adaptive Boosting) algorithm for both classification and regression tasks.

## 📚 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [1. Holiday Package Purchase Prediction (Classification)](#1-holiday-package-purchase-prediction-classification)
  - [2. Used Car Price Prediction (Regression)](#2-used-car-price-prediction-regression)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

AdaBoost (Adaptive Boosting) is a powerful ensemble learning method that combines multiple weak learners to create a strong classifier or regressor. This repository showcases two real-world applications:

1. **Classification**: Predicting whether a customer will purchase a holiday package
2. **Regression**: Predicting the selling price of used cars

Both projects follow a complete machine learning pipeline from data collection to model evaluation, demonstrating best practices in data preprocessing, feature engineering, model training, and performance optimization.

## 📊 Projects

### 1. Holiday Package Purchase Prediction (Classification)

**Business Context:**
"Trips & Travel.Com" aims to optimize marketing expenditure by targeting potential customers for their new Wellness Tourism Package. Currently, only 18% of randomly contacted customers purchase packages, resulting in high marketing costs.

**Objective:**
Build a classification model to predict which customers are likely to purchase the new wellness tourism package, enabling targeted marketing campaigns.

**Dataset:**
- **Source:** [Kaggle - Holiday Package Purchase Prediction](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)
- **Size:** 4,888 rows × 20 columns
- **Target Variable:** `ProdTaken` (1 = purchased, 0 = not purchased)

**Key Features:**
- CustomerID
- Age
- TypeofContact (Self Enquiry / Company Invited)
- CityTier
- DurationOfPitch
- Occupation
- Gender
- NumberOfPersonVisiting
- NumberOfFollowups
- ProductPitched (Basic, Standard, Deluxe, Super Deluxe, King)
- PreferredPropertyStar
- MaritalStatus
- NumberOfTrips
- Passport
- PitchSatisfactionScore
- OwnCar
- NumberOfChildrenVisiting
- Designation
- MonthlyIncome

**Data Challenges:**
- Missing values in multiple columns (Age, TypeofContact, DurationOfPitch, etc.)
- Imbalanced target variable (18% positive class)
- Mix of numerical and categorical features
- Outliers in continuous variables

### 2. Used Car Price Prediction (Regression)

**Business Context:**
Predict the selling price of used cars based on various features to help buyers and sellers make informed decisions in the automotive marketplace.

**Objective:**
Develop a regression model that accurately estimates the market value of used cars based on their characteristics.

**Dataset:**
- Used car listings with features including make, model, year, mileage, and condition
- Target Variable: Selling price of the vehicle

**Key Features:**
- Vehicle specifications (make, model, year)
- Mileage
- Engine specifications
- Transmission type
- Fuel type
- Previous ownership history
- Vehicle condition metrics

**Data Challenges:**
- Missing values requiring imputation
- Categorical variables needing encoding
- Feature scaling requirements
- Potential outliers in pricing data

## ✨ Key Features

Both projects implement:

- **Comprehensive Data Cleaning:**
  - Missing value handling with appropriate imputation strategies
  - Duplicate detection and removal
  - Data type verification and conversion
  - Outlier detection and treatment

- **Exploratory Data Analysis (EDA):**
  - Statistical summaries
  - Distribution analysis
  - Correlation studies
  - Feature relationship visualization

- **Feature Engineering:**
  - One-hot encoding for categorical variables
  - Label encoding for ordinal features
  - Feature scaling and normalization
  - Feature selection and dimensionality reduction

- **Model Development:**
  - AdaBoost classifier/regressor implementation
  - Hyperparameter tuning
  - Cross-validation
  - Model comparison with baseline algorithms

- **Performance Evaluation:**
  - **Classification Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - **Regression Metrics:** MAE, MSE, RMSE, R² Score
  - Confusion matrix analysis
  - Feature importance analysis

## 🛠️ Technologies Used

- **Python 3.x**
- **Core Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical data visualization
  
- **Machine Learning:**
  - `scikit-learn` - Machine learning algorithms and utilities
    - `AdaBoostClassifier`
    - `AdaBoostRegressor`
    - Preprocessing utilities
    - Model evaluation metrics

- **Development Environment:**
  - Jupyter Notebook
  - Python virtual environment

## 📁 Project Structure

```
.
├── AdaboostClassification.ipynb    # Holiday package prediction notebook
├── AdaboostRegression.ipynb        # Used car price prediction notebook
├── README.md                       # Project documentation
└── data/
    ├── Travel.csv                  # Holiday package dataset (if included)
    └── used_cars.csv              # Used car dataset (if included)
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7 or higher
pip package manager
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd adaboost-ml-projects
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

5. Open either `AdaboostClassification.ipynb` or `AdaboostRegression.ipynb`

### Running the Notebooks

1. Ensure your dataset is in the correct location
2. Run all cells sequentially from top to bottom
3. Review outputs, visualizations, and model performance metrics
4. Experiment with different hyperparameters or preprocessing techniques

## 📈 Methodology

Both projects follow a standardized machine learning pipeline:

### 1. Data Collection & Understanding
- Import datasets from reliable sources
- Understand business context and problem statement
- Examine data structure and basic statistics

### 2. Data Cleaning
- Handle missing values (imputation, deletion)
- Remove duplicates
- Convert data types
- Identify and treat outliers

### 3. Exploratory Data Analysis
- Univariate analysis (distributions, frequencies)
- Bivariate analysis (relationships with target)
- Multivariate analysis (feature interactions)
- Visualization of key patterns

### 4. Feature Engineering
- Encode categorical variables
  - One-hot encoding for nominal features
  - Label encoding for ordinal features
- Scale/normalize numerical features
- Create derived features (if applicable)
- Feature selection

### 5. Train-Test Split
- Split data into training and testing sets
- Maintain stratification for classification
- Ensure representative samples

### 6. Model Training
- Initialize AdaBoost algorithm
- Configure hyperparameters
- Fit model on training data
- Implement cross-validation

### 7. Model Evaluation
- Make predictions on test set
- Calculate performance metrics
- Analyze confusion matrix (classification)
- Examine residuals (regression)
- Feature importance analysis

### 8. Model Optimization
- Hyperparameter tuning (grid search, random search)
- Compare with other algorithms
- Ensemble methods
- Final model selection

## 📊 Results

### Classification Project (Holiday Package Prediction)

**Key Insights:**
- Successfully identified key factors influencing package purchases
- Achieved improved prediction accuracy over random selection
- Reduced marketing costs through targeted customer identification
- Feature importance revealed critical decision-making factors

**Business Impact:**
- More efficient marketing spend allocation
- Higher conversion rates
- Better customer targeting strategy
- Data-driven decision making for product launches

### Regression Project (Used Car Price Prediction)

**Key Insights:**
- Developed accurate price prediction model
- Identified key value drivers in used car market
- Quantified impact of various vehicle features on pricing
- Created interpretable model for stakeholder understanding

**Business Impact:**
- Fair pricing recommendations for buyers and sellers
- Market value estimation for inventory management
- Data-driven pricing strategy
- Reduced pricing discrepancies

## 🎓 Learning Outcomes

This project demonstrates:

1. **AdaBoost Algorithm Understanding:**
   - How AdaBoost combines weak learners
   - Difference between classification and regression applications
   - Boosting vs. bagging techniques

2. **Complete ML Pipeline:**
   - End-to-end project execution
   - Data preprocessing best practices
   - Model evaluation strategies

3. **Real-World Applications:**
   - Business problem formulation
   - Data-driven decision making
   - Model interpretation and communication

4. **Technical Skills:**
   - Python programming for data science
   - scikit-learn library proficiency
   - Data visualization techniques
   - Model performance optimization

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

**Areas for Contribution:**
- Additional feature engineering techniques
- Alternative algorithms for comparison
- Enhanced visualizations
- Documentation improvements
- Code optimization
- Additional datasets

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the repository.

---

## 🔗 Additional Resources

- [AdaBoost Algorithm Documentation](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Dataset - Holiday Package](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)
- [Machine Learning Mastery - AdaBoost](https://machinelearningmastery.com/adaboost-ensemble-in-python/)

## 🙏 Acknowledgments

- Kaggle community for providing quality datasets
- scikit-learn developers for excellent ML library
- Open-source community for tools and resources

---

**Note:** This project is for educational and demonstration purposes. Always ensure you have the right to use datasets and follow data privacy regulations when working with real customer data.
