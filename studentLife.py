import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Read the data
df = pd.read_csv('student_lifestyle_dataset.csv')

def explore_data(df):
    """Perform initial data exploration"""
    print("\nDataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nUnique values in categorical columns:")
    print(df['Stress_Level'].value_counts())


def create_visualizations(df):
    """Create basic visualizations of the data"""
    # Figure 1: Basic Metrics
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
    fig1.suptitle('Student Performance Analysis - Basic Metrics', y=1.02)
    
    # Plot 1: Distribution of GPA
    sns.histplot(data=df, x='GPA', ax=axes1[0,0])
    axes1[0,0].set_title('Distribution of GPA')
    
    # Plot 2: Study Hours vs GPA
    sns.scatterplot(data=df, x='Study_Hours_Per_Day', y='GPA', ax=axes1[0,1])
    axes1[0,1].set_title('Study Hours vs GPA')
    
    # Plot 3: Sleep Hours vs GPA
    sns.scatterplot(data=df, x='Sleep_Hours_Per_Day', y='GPA', ax=axes1[1,0])
    axes1[1,0].set_title('Sleep Hours vs GPA')
    
    # Plot 4: Stress Level vs GPA Scatter
    sns.scatterplot(data=df, x='Student_ID', y='GPA', hue='Stress_Level', ax=axes1[1,1])
    axes1[1,1].set_title('GPA by Stress Level')
    axes1[1,1].legend(title='Stress Level')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Stress Analysis
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Stress Level Impact on GPA', y=1.02)
    
    # Plot 5: Stress Level Distribution
    sns.countplot(data=df, x='Stress_Level', ax=ax1)
    ax1.set_title('Distribution of Stress Levels')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 6: Stress Level vs GPA (Box Plot)
    sns.boxplot(data=df, x='Stress_Level', y='GPA', ax=ax2)
    ax2.set_title('GPA Distribution by Stress Level')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 7: Stress Level vs GPA (Violin Plot)
    sns.violinplot(data=df, x='Stress_Level', y='GPA', ax=ax3)
    ax3.set_title('Detailed GPA Distribution by Stress Level')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

    print("\nAverage GPA by Stress Level:")
    print(df.groupby('Stress_Level')['GPA'].mean().sort_values(ascending=False))
    
    print("\nNumber of Students in Each Stress Level:")
    print(df['Stress_Level'].value_counts())

def analyze_correlations(df):
    """Analyze correlations between numerical variables"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def preprocess_data(df):
    """Complete data preprocessing steps"""
    print("\n=== Data Preprocessing ===")
    
    # 1. Check and handle missing values
    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())
    
    # 2. Check and handle outliers using IQR method
    def handle_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Apply outlier handling to numerical columns
    numerical_columns = ['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
                        'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 
                        'Physical_Activity_Hours_Per_Day', 'GPA']
    
    df_cleaned = df.copy()
    for column in numerical_columns:
        df_cleaned = handle_outliers(df_cleaned, column)
    
    print("\nRows before outlier removal:", len(df))
    print("Rows after outlier removal:", len(df_cleaned))
    
    # 3. Feature Engineering
    df_cleaned['Total_Activity_Hours'] = df_cleaned['Study_Hours_Per_Day'] + \
                                       df_cleaned['Extracurricular_Hours_Per_Day'] + \
                                       df_cleaned['Physical_Activity_Hours_Per_Day']
    
    df_cleaned['Study_Sleep_Ratio'] = df_cleaned['Study_Hours_Per_Day'] / df_cleaned['Sleep_Hours_Per_Day']
    
    df_cleaned['Activity_Balance'] = df_cleaned['Study_Hours_Per_Day'] / df_cleaned['Total_Activity_Hours']
    
    # 4. Standardize numerical features
    scaler = StandardScaler()
    df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])
    
    print("\nNew features added:")
    print("- Total_Activity_Hours")
    print("- Study_Sleep_Ratio")
    print("- Activity_Balance")
    
    return df_cleaned


# Update your existing prepare_data function with this new version:
def prepare_data(df):
    """Prepare features and target for modeling"""
    # Preprocess the data
    df_preprocessed = preprocess_data(df)
    
    # Prepare features (X) and target (y), including new engineered features
    X = df_preprocessed[['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
                        'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 
                        'Physical_Activity_Hours_Per_Day', 'Total_Activity_Hours',
                        'Study_Sleep_Ratio', 'Activity_Balance']]
    y = df_preprocessed['GPA']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest and Gradient Boosting models"""
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    return rf_model, gb_model, rf_pred, gb_pred

def cross_validate_models(X, y, models):
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"\n{name} Cross-Validation Scores:")
        print(f"Mean R2: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print model performance metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

#Prediction vs Actual Plot
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual GPA')
    plt.ylabel('Predicted GPA')
    plt.title(f'{model_name}: Predicted vs Actual GPA')
    plt.tight_layout()
    plt.show()

def plot_model_comparison(y_test, rf_pred, gb_pred):
    """Plot comparison of model predictions"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.5)
    plt.scatter(range(len(rf_pred)), rf_pred, label='Random Forest', alpha=0.5)
    plt.scatter(range(len(gb_pred)), gb_pred, label='Gradient Boosting', alpha=0.5)
    plt.title('Model Predictions Comparison')
    plt.xlabel('Sample')
    plt.ylabel('GPA')
    plt.legend()
    
    # Plot 2: Error Distribution
    plt.subplot(1, 2, 2)
    plt.hist(y_test - rf_pred, alpha=0.5, label='Random Forest Error', bins=20)
    plt.hist(y_test - gb_pred, alpha=0.5, label='Gradient Boosting Error', bins=20)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_regression_results(y_test, predictions, model_names):
    """Plot regression results for multiple models"""
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(15, 5))
    fig.suptitle('Model Predictions vs Actual Values')
    
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        axes[i].scatter(y_test, pred, alpha=0.5)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[i].set_xlabel('Actual GPA')
        axes[i].set_ylabel('Predicted GPA')
        axes[i].set_title(f'{name}')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, features, model_name):
    """Plot feature importance for a model"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8, 4))
    plt.title(f'Feature Importance ({model_name})')
    plt.bar(range(features.shape[1]), importances[indices])
    plt.xticks(range(features.shape[1]), 
               [features.columns[i].replace('_', ' ') for i in indices], 
               rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Read the data
    df = pd.read_csv('student_lifestyle_dataset.csv')
    
    # Data exploration
    explore_data(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Prepare the data and get all necessary variables
    X_train, X_test, y_train, y_test, X = prepare_data(df)
    y = df['GPA']  # Add this line
    
    # Train and evaluate models
    rf_model, gb_model, rf_pred, gb_pred = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n=== Model Evaluation and Visualization ===")
    
    # Model performance metrics
    evaluate_model(y_test, rf_pred, "Random Forest")
    evaluate_model(y_test, gb_pred, "Gradient Boosting")
    
    # Feature importance
    plot_feature_importance(rf_model, X, "Random Forest")
    plot_feature_importance(gb_model, X, "Gradient Boosting")
    
    # Model comparison visualizations
    plot_model_comparison(y_test, rf_pred, gb_pred)
    plot_regression_results(y_test, [rf_pred, gb_pred], ["Random Forest", "Gradient Boosting"])