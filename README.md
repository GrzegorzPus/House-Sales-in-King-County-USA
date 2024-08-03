# 🏠 House Sales in King County, USA

Welcome to my GitHub repository dedicated to analyzing house sales in King County, USA! This repository contains two projects focusing on different aspects of the housing data analysis. Below, you will find detailed descriptions of the projects along with links to the analysis files, relevant code snippets, and visualizations. Dive in to explore the insights from King County's housing market!
<br>
<br>
## 📈 Project 1: [Machine Learning Model Comparison for House Price Prediction](https://github.com/GrzegorzPus/House-Sales-in-King-County-USA/blob/main/Predictive%20Models.ipynb)

This project involves building and comparing various machine learning models to predict house prices in King County, USA. The analysis includes the following steps:

1. **Data Preparation**:
   - Cleaned and transformed the housing data to make it suitable for model building.

2. **Model Building**:
   - Implemented multiple machine learning models including Linear Regression, Polynomial Regression, Decision Trees, Random Forests.

3. **Model Evaluation**:
   - Evaluated the performance of each model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R^2).

4. **Model Comparison**:
   - Compared the models to determine the best-performing one based on evaluation metrics.
<br>

### 📊 Example Visualization

![image](https://github.com/user-attachments/assets/259dea3c-62ce-4b3b-887d-f4fc65c7fd62)
*Comparison of different machine learning models*
<br>
<br>

   ### 🔑 Key Findings
   Some of the key determinants analyzed include:
   - **Best Model**: Random Forest outperformed other models with the lowest MAE and highest R-squared value.
<br>

   ### 🛠️ Technologies
   - **Programming Languages**: Python
  - **Python**: `pandas`  `numpy` `matplotlib` `statsmodels` `sklearn`
<br>

  ### 👨‍💻 Sample Code
   ```python
   from sklearn.preprocessing import PolynomialFeatures
  polnomialy_regression = PolynomialFeatures(degree = 3)
  x_poly = polnomialy_regression.fit_transform(X)
  linear_regression2 = LinearRegression()
  linear_regression2.fit(x_poly,y)

  print(f'R² score: {r2_score(Y, linear_regression2.predict(polnomialy_regression.fit_transform(X)))*100}')
   ```
---
<br>

## 📈 Project 2: [Machine Learning Model Comparison for House Price Prediction](https://github.com/GrzegorzPus/House-Sales-in-King-County-USA/blob/main/House%20Sales%20-%20Data%20Exploration.ipynb)

This project involves performing a comprehensive exploratory data analysis (EDA) of the house sales data in King County, USA. The goal of this analysis is to understand the data, identify key trends and patterns, and derive insights that can inform further modeling and decision-making. The analysis includes the following steps:

1. **Data Cleaning**:
   - **Missing Values**: Removed or imputed missing values.
   - **Outliers**: Identified and handled outliers to avoid skewing the analysis.
   - **Data Types**: Ensured all columns had appropriate data types for analysis.

2. **Descriptive Statistics**:
   - **Measures of Central Tendency**: Mean, median, and mode of key variables.
   - **Dispersion Metrics**: Standard deviation, variance, and range.

3. **Visualization of Key Variables**:
   - **Price Distribution**: Visualized the distribution of house prices to understand the range and common price points.
   - **Scatter Plots**: Plotted house prices against features like square footage and number of bedrooms.
   - **Box Plots**: Used to show the spread and skewness of price data across different categories.
<br>

### 📊 Example Visualization

![image](https://github.com/user-attachments/assets/abf96968-0e05-4d57-b088-5d6023f48dd0)
*Comparison of average prices by zipcode*
<br>
<br>

   ### 🔑 Key Findings
   Some of the key determinants analyzed include:
   - **Price Distribution**: Most houses are priced bellow \$1,000,000 with a few high-value outliers.
   - **Key Predictors**: Square footage, location, and number of bedrooms are significant predictors of house prices.
<br>

   ### 🛠️ Technologies
   - **Programming Languages**: Python
  - **Python**: `pandas`  `numpy` `matplotlib` `seaborn`
<br>

  ### 👨‍💻 Sample Code
   ```python
  plt.figure(figsize=(14, 6))
  sns.countplot(x='bedrooms', data=df, edgecolor='black')
  plt.xlabel('Number of Bedrooms')
  plt.ylabel('Count')
  plt.title('Number of Bedrooms')
  ax = plt.gca()
  for p in ax.patches:
      ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

  plt.show()
   ```
---
<br>

## 👥 Contact

If you have any questions about the projects or would like to discuss potential collaborations, feel free to reach out:

- **[LinkedIn](https://www.linkedin.com/in/grzegorz-pu%C5%9B/)**

Thank you for visiting my repository!

---

#Python #DataAnalysis #MachineLearning #LinearRegression
