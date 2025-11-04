# For Data Manipulation & Numerical Operations
import pandas as pd
import numpy as np

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

# For Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# For Statistical Analysis
import statsmodels.api as sm

# To Load the dataset
df = pd.read_csv('D:/data_insurance.csv')

# For Initial data inspection
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())

# Display basic descriptive statistics for numerical columns
print("Descriptive Statistics for Numerical Variables:")
print(df.describe())

# Display value counts for categorical variables
print("\nValue Counts for Categorical Variables:")
print("Sex distribution:")
print(df['sex'].value_counts())
print("\nSmoker distribution:")
print(df['smoker'].value_counts())
print("\nRegion distribution:")
print(df['region'].value_counts())

# Check for missing values
print("\nMissing Values Check:")
print(df.isnull().sum())

# Set up the visualization style
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Histogram of charges
ax1.hist(df['charges'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_title('Distribution of Medical Charges')
ax1.set_xlabel('Charges ($)')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# Boxplot of charges
ax2.boxplot(df['charges'])
ax2.set_title('Boxplot of Medical Charges')
ax2.set_ylabel('Charges ($)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate skewness
skewness = df['charges'].skew()
print(f"Skewness of charges: {skewness:.4f}")
print(f"Mean charges: ${df['charges'].mean():.2f}")
print(f"Median charges: ${df['charges'].median():.2f}")

# Set up the figure for subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Age distribution
axes[0, 0].hist(df['age'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
axes[0, 0].set_title('Distribution of Age')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: BMI distribution
axes[0, 1].hist(df['bmi'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribution of BMI')
axes[0, 1].set_xlabel('BMI')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(18.5, color='red', linestyle='--', alpha=0.7, label='Underweight threshold')
axes[0, 1].axvline(24.9, color='green', linestyle='--', alpha=0.7, label='Healthy threshold')
axes[0, 1].axvline(29.9, color='orange', linestyle='--', alpha=0.7, label='Overweight threshold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Children count distribution
children_counts = df['children'].value_counts().sort_index()
axes[1, 0].bar(children_counts.index, children_counts.values, alpha=0.7, color='mediumpurple', edgecolor='black')
axes[1, 0].set_title('Distribution of Number of Children')
axes[1, 0].set_xlabel('Number of Children')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Smoker distribution
smoker_counts = df['smoker'].value_counts()
axes[1, 1].bar(smoker_counts.index, smoker_counts.values, alpha=0.7, color='gold', edgecolor='black')
axes[1, 1].set_title('Distribution of Smoker Status')
axes[1, 1].set_xlabel('Smoker')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print key statistics
print(f"Average age: {df['age'].mean():.1f} years")
print(f"Average BMI: {df['bmi'].mean():.1f}")
print(f"Percentage of smokers: {(df['smoker'] == 'yes').mean() * 100:.1f}%")

# Set up the figure for subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Age vs Charges (colored by smoker status)
scatter1 = axes[0, 0].scatter(df['age'], df['charges'], c=df['smoker'].map({'yes': 1, 'no': 0}), 
                             alpha=0.6, cmap='viridis')
axes[0, 0].set_title('Age vs Charges (Color: Smoker Status)')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Charges ($)')
axes[0, 0].grid(True, alpha=0.3)
legend1 = axes[0, 0].legend(*scatter1.legend_elements(), title="Smoker")
axes[0, 0].add_artist(legend1)

# Plot 2: BMI vs Charges (colored by smoker status)
scatter2 = axes[0, 1].scatter(df['bmi'], df['charges'], c=df['smoker'].map({'yes': 1, 'no': 0}), 
                             alpha=0.6, cmap='viridis')
axes[0, 1].set_title('BMI vs Charges (Color: Smoker Status)')
axes[0, 1].set_xlabel('BMI')
axes[0, 1].set_ylabel('Charges ($)')
axes[0, 1].grid(True, alpha=0.3)
legend2 = axes[0, 1].legend(*scatter2.legend_elements(), title="Smoker")
axes[0, 1].add_artist(legend2)

# Plot 3: Boxplot of Charges by Smoker Status
df.boxplot(column='charges', by='smoker', ax=axes[1, 0])
axes[1, 0].set_title('Charges by Smoker Status')
axes[1, 0].set_xlabel('Smoker')
axes[1, 0].set_ylabel('Charges ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Boxplot of Charges by Number of Children
df.boxplot(column='charges', by='children', ax=axes[1, 1])
axes[1, 1].set_title('Charges by Number of Children')
axes[1, 1].set_xlabel('Number of Children')
axes[1, 1].set_ylabel('Charges ($)')

plt.suptitle('')  # Remove automatic subtitle
plt.tight_layout()
plt.show()

# Calculate correlation matrix for numerical variables
numeric_df = df[['age', 'bmi', 'children', 'charges']]
correlation_matrix = numeric_df.corr()
print("Correlation Matrix (Numerical Variables):")
print(correlation_matrix)

# Create new features based on insights from EDA
df['age_squared'] = df['age'] ** 2  # Capture potential non-linear relationship with age

# Create interaction term: BMI * Smoker
df['bmi_smoker_interaction'] = df['bmi'] * (df['smoker'] == 'yes').astype(int)

# Create another interaction: Age * Smoker
df['age_smoker_interaction'] = df['age'] * (df['smoker'] == 'yes').astype(int)

# Verify the new features
print("New features created:")
print(f"- age_squared: range {df['age_squared'].min()} to {df['age_squared'].max()}")
print(f"- bmi_smoker_interaction: {df['bmi_smoker_interaction'].sum():.0f} non-zero values (smokers)")
print(f"- age_smoker_interaction: {df['age_smoker_interaction'].sum():.0f} non-zero values (smokers)")

# Show first 5 rows with new features
print("\nSample data with new features:")
print(df[['age', 'age_squared', 'bmi', 'smoker', 'bmi_smoker_interaction', 'age_smoker_interaction']].head(10))

# Quick check of correlation with charges for new features
new_features_corr = df[['age_squared', 'bmi_smoker_interaction', 'age_smoker_interaction', 'charges']].corr()
print("\nCorrelation of new features with charges:")
print(new_features_corr['charges'].sort_values(ascending=False))

# Define the features (X) and target (y)
X = df[['age', 'age_squared', 'bmi', 'children', 'bmi_smoker_interaction', 'age_smoker_interaction', 'sex', 'smoker', 'region']]
y = df['charges']

# Display the shape and feature names
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFeature names:")
print(X.columns.tolist())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")

# Recreate the preprocessing pipeline (to ensure all variables are defined)
numerical_features = ['age', 'age_squared', 'bmi', 'children', 'bmi_smoker_interaction', 'age_smoker_interaction']
categorical_features = ['sex', 'smoker', 'region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Now train the model on the training data
model.fit(X_train, y_train)

# Display training completion message
print("Model training completed successfully!")
print(f"Model trained on {X_train.shape[0]} samples with {X_train.shape[1]} features")

# Check if the model has been fitted
print(f"Model is fitted: {model.named_steps['regressor'].coef_ is not None}")

# Get the number of coefficients
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
print(f"\nNumber of coefficients in the model: {len(model.named_steps['regressor'].coef_)}")
print(f"Number of feature names after preprocessing: {len(feature_names)}")

# Show the intercept term
intercept = model.named_steps['regressor'].intercept_
print(f"Model intercept: ${intercept:,.2f}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display performance metrics
print(" Model Performance Evaluation:")
print(f"Mean Squared Error (MSE): ${mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R²) Score: {r2:.4f}")

# Create actual vs predicted scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Charges ($)')
plt.ylabel('Predicted Charges ($)')
plt.title('Actual vs Predicted Medical Charges')
plt.grid(True, alpha=0.3)

# Add RMSE and R² to plot
plt.text(0.05, 0.95, f'RMSE = ${rmse:,.2f}\nR² = {r2:.4f}', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Compare actual vs predicted for first 10 test samples
print("\n Sample Predictions (First 10 test samples):")
comparison = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Difference': y_test.values[:10] - y_pred[:10]
})
comparison['Actual'] = comparison['Actual'].apply(lambda x: f"${x:,.2f}")
comparison['Predicted'] = comparison['Predicted'].apply(lambda x: f"${x:,.2f}")
comparison['Difference'] = comparison['Difference'].apply(lambda x: f"${x:,.2f}")
print(comparison)

# Get feature names and coefficients from the trained model
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coefficients = model.named_steps['regressor'].coef_

# Create a DataFrame for clear analysis
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Effect': np.abs(coefficients)  # Magnitude of effect
}).sort_values('Abs_Effect', ascending=False)

print("Top 10 Most Influential Factors on Insurance Costs:")
print(coef_df[['Feature', 'Coefficient']].head(10))

print("\n" + "="*60)
print("Interpretation Guide:")
print("Positive coefficient = Higher feature value corresponds to Higher cost")
print("Negative coefficient = Higher feature value corresponds to Lower cost")
print("="*60)

# Plot the coefficients
plt.figure(figsize=(14, 10))
colors = ['red' if coef < 0 else 'green' for coef in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
plt.xlabel('Coefficient Magnitude (Impact on Charges)', fontsize=12)
plt.title('Feature Impact on Medical Insurance Costs\n(Green = Increases Cost, Red = Decreases Cost)', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Detailed interpretation of key features
print("\n Key Insights from Coefficient Analysis:")
print("1. Smoker Status: The strongest predictor by far")
print("2. Age Effects: Both linear (age) and quadratic (age_squared) terms matter")
print("3. BMI Impact: Significant effect, especially when combined with smoking")
print("4. Demographic Factors: Region and sex have smaller but meaningful effects")

print("\n" + "="*65)
print("MODEL PERFORMANCE SUMMARY:")
print(f"R² = {r2:.4f} → Model explains {r2*100:.1f}% of variance in medical charges")
print(f"RMSE = ${rmse:,.2f} → Average prediction error")
print(f"MSE = ${mse:,.2f} → Mean squared error")