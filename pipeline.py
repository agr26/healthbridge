import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging

class HealthcareAnalyticsPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self.models = {}
        self.scalers = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the analytics pipeline."""
        logger = logging.getLogger('HealthcareAnalytics')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('healthcare_analytics.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_and_preprocess_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess healthcare data."""
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Handle missing values
            df['age'] = df['age'].fillna(df['age'].median())
            df['bmi'] = df['bmi'].fillna(df['bmi'].median())
            
            # Create features
            df['length_of_stay'] = (pd.to_datetime(df['discharge_date']) - 
                                  pd.to_datetime(df['admission_date'])).dt.days
            
            # Extract temporal features
            df['admission_hour'] = pd.to_datetime(df['admission_date']).dt.hour
            df['admission_dow'] = pd.to_datetime(df['admission_date']).dt.dayofweek
            
            # Create target variables
            y_readmission = df['readmitted_30_days']
            
            # Drop unnecessary columns
            X = df.drop(['readmitted_30_days', 'patient_id', 'admission_date', 
                        'discharge_date'], axis=1)
            
            return X, y_readmission
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def train_readmission_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the readmission prediction model."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            self.logger.info(f"Readmission Model AUC-ROC: {auc_roc}")
            
            # Store model and scaler
            self.models['readmission'] = model
            self.scalers['readmission'] = scaler
            
        except Exception as e:
            self.logger.error(f"Error in training readmission model: {str(e)}")
            raise

    def analyze_social_determinants(self, df: pd.DataFrame) -> Dict:
        """Analyze impact of social determinants of health."""
        results = {}
        
        try:
            # Analyze correlation with outcomes
            sdoh_cols = ['education_level', 'income_level', 'housing_status', 
                        'transportation_access']
            
            for col in sdoh_cols:
                correlation = df[col].corr(df['readmitted_30_days'])
                results[f'{col}_correlation'] = correlation
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            sns.heatmap(df[sdoh_cols + ['readmitted_30_days']].corr(), 
                       annot=True, cmap='coolwarm')
            plt.title('SDOH Correlations with Readmission')
            plt.savefig('sdoh_correlations.png')
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in SDOH analysis: {str(e)}")
            raise

    def create_neural_network(self, input_shape: int) -> tf.keras.Model:
        """Create neural network for complex pattern recognition."""
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        
        return model

    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in healthcare data."""
        results = {}
        
        try:
            # Analyze daily patterns
            daily_admissions = df.groupby('admission_hour')['readmitted_30_days'].mean()
            results['hourly_readmission_risk'] = daily_admissions.to_dict()
            
            # Analyze seasonal patterns
            df['admission_month'] = pd.to_datetime(df['admission_date']).dt.month
            monthly_patterns = df.groupby('admission_month').agg({
                'readmitted_30_days': 'mean',
                'length_of_stay': 'mean'
            }).to_dict()
            
            results['monthly_patterns'] = monthly_patterns
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {str(e)}")
            raise

    def generate_patient_risk_profile(self, patient_data: Dict) -> Dict:
        """Generate comprehensive patient risk profile."""
        try:
            risk_factors = []
            risk_score = 0.0
            
            # Clinical factors
            if patient_data['age'] > 65:
                risk_factors.append('Advanced Age')
                risk_score += 0.2
                
            if patient_data['previous_admissions'] > 2:
                risk_factors.append('Multiple Previous Admissions')
                risk_score += 0.3
                
            if patient_data['chronic_conditions'] > 3:
                risk_factors.append('Multiple Chronic Conditions')
                risk_score += 0.25
                
            # Social factors
            if patient_data['transportation_access'] == False:
                risk_factors.append('Limited Transportation Access')
                risk_score += 0.15
                
            if patient_data['social_support_score'] < 3:
                risk_factors.append('Low Social Support')
                risk_score += 0.1
                
            return {
                'risk_score': min(risk_score, 1.0),
                'risk_factors': risk_factors,
                'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.4 else 'Low'
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk profile generation: {str(e)}")
            raise