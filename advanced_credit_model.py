import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

warnings.filterwarnings('ignore')

# Set up logging for professional tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CreditScoringPipeline:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.best_model = None
        self.preprocessor = None
        
        # Ensure output directories exist
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('models', exist_ok=True)

    def generate_mock_data(self, n_samples=1000):
        """Generates mock Indian banking data if no dataset is provided."""
        logger.info("Generating mock data for the pipeline...")
        np.random.seed(42)
        
        data = {
            'Age': np.random.randint(21, 60, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Marital_Status': np.random.choice(['Single', 'Married'], n_samples),
            'Education': np.random.choice(['Graduate', 'Post-Graduate', 'Under-Graduate'], n_samples),
            'Employment_Type': np.random.choice(['Salaried', 'Self-Employed'], n_samples),
            'Monthly_Income': np.random.randint(20000, 250000, n_samples),
            'Loan_Amount': np.random.randint(50000, 5000000, n_samples),
            'Loan_Tenure_Months': np.random.randint(12, 120, n_samples),
            'Number_of_Credit_Cards': np.random.randint(0, 5, n_samples),
            'Delayed_Payments': np.random.randint(0, 10, n_samples),
            'CIBIL_Score': np.random.randint(300, 900, n_samples),
            'Creditworthy': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]) # 30% default rate
        }
        
        self.df = pd.DataFrame(data)
        # Feature Engineering: EMI approximation & Debt-to-Income
        self.df['EMI_Amount'] = self.df['Loan_Amount'] / self.df['Loan_Tenure_Months'] * 1.1 
        self.df['Debt_to_Income_Ratio'] = self.df['EMI_Amount'] / self.df['Monthly_Income']
        
        logger.info(f"Generated {n_samples} records.")
        return self.df

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def perform_eda(self):
        """Generates EDA visualizations and saves them to the outputs folder."""
        logger.info("Starting Exploratory Data Analysis...")
        
        # 1. Class Imbalance Check
        plt.figure(figsize=(6,4))
        sns.countplot(x='Creditworthy', data=self.df)
        plt.title('Target Variable Distribution')
        plt.savefig('outputs/class_distribution.png')
        plt.close()

        # 2. CIBIL Score vs Creditworthiness
        plt.figure(figsize=(8,5))
        sns.boxplot(x='Creditworthy', y='CIBIL_Score', data=self.df)
        plt.title('CIBIL Score vs Credit Risk')
        plt.savefig('outputs/cibil_vs_risk.png')
        plt.close()

        # 3. Correlation Heatmap (Numerical features only)
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        plt.figure(figsize=(12,8))
        sns.heatmap(self.df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Matrix')
        plt.savefig('outputs/correlation_matrix.png')
        plt.close()
        
        logger.info("EDA completed. Plots saved to 'outputs/' directory.")

    def preprocess_data(self):
        logger.info("Preprocessing data...")
        X = self.df.drop('Creditworthy', axis=1)
        y = self.df['Creditworthy']

        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object']).columns

        # Pipelines for numerical and categorical features
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        logger.info("Training multiple models...")
        
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'XGBoost': XGBClassifier(scale_pos_weight=1, random_state=42) # Adjust scale_pos_weight for imbalance
        }

        best_score = 0
        results = {}

        for name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {'ROC-AUC': roc_auc, 'F1-Score': f1}
            logger.info(f"{name} - ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}")

            # Keep track of the best model based on ROC-AUC
            if roc_auc > best_score:
                best_score = roc_auc
                self.best_model = pipeline
                self.best_model_name = name

        logger.info(f"Best Model Selected: {self.best_model_name} with ROC-AUC: {best_score:.4f}")

    def hyperparameter_tuning(self, X_train, y_train):
        """Demonstrates tuning on the Random Forest model."""
        logger.info("Performing Hyperparameter Tuning on Random Forest...")
        
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])

        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }

        search = RandomizedSearchCV(rf_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, n_iter=5, random_state=42)
        search.fit(X_train, y_train)
        
        logger.info(f"Best Parameters: {search.best_params_}")
        self.best_model = search.best_estimator_ # Update best model

    def save_model(self):
        try:
            joblib.dump(self.best_model, 'models/credit_risk_model.joblib')
            logger.info("Model saved to 'models/credit_risk_model.joblib'")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def predict(self, applicant_data):
        """Inference function for a new applicant."""
        if not self.best_model:
            self.best_model = joblib.load('models/credit_risk_model.joblib')
            
        df_input = pd.DataFrame([applicant_data])
        
        # Engineering the same features as training
        if 'EMI_Amount' not in df_input.columns:
            df_input['EMI_Amount'] = df_input['Loan_Amount'] / df_input['Loan_Tenure_Months'] * 1.1
        if 'Debt_to_Income_Ratio' not in df_input.columns:
            df_input['Debt_to_Income_Ratio'] = df_input['EMI_Amount'] / df_input['Monthly_Income']

        prob = self.best_model.predict_proba(df_input)[0][1]
        prediction = self.best_model.predict(df_input)[0]
        
        category = "Good Credit Risk" if prediction == 1 else "Bad Credit Risk"
        recommendation = "Approve" if prediction == 1 else "Reject/Review manually"

        return {
            "Credit Risk Category": category,
            "Approval Recommendation": recommendation,
            "Probability Score (0-1)": round(prob, 4)
        }

if __name__ == "__main__":
    # Orchestration
    pipeline = CreditScoringPipeline()
    
    # 1. Load or Generate Data
    pipeline.generate_mock_data(n_samples=2000) 
    
    # 2. EDA
    pipeline.perform_eda()
    
    # 3. Preprocessing
    X_train, X_test, y_train, y_test = pipeline.preprocess_data()
    
    # 4. Model Training & Comparison
    pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 5. Tuning (Optional, focusing on RF for this example)
    pipeline.hyperparameter_tuning(X_train, y_train)
    
    # 6. Save Artifacts
    pipeline.save_model()
    
    # 7. Mock Inference (System test)
    sample_applicant = {
        'Age': 28, 'Gender': 'Male', 'Marital_Status': 'Single', 
        'Education': 'Graduate', 'Employment_Type': 'Salaried',
        'Monthly_Income': 85000, 'Loan_Amount': 500000, 'Loan_Tenure_Months': 60,
        'Number_of_Credit_Cards': 2, 'Delayed_Payments': 0, 'CIBIL_Score': 780
    }
    
    print("\n--- Prediction for New Applicant ---")
    result = pipeline.predict(sample_applicant)
    for key, val in result.items():
        print(f"{key}: {val}")