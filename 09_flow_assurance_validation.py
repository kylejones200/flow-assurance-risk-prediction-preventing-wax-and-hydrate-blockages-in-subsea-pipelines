#!/usr/bin/env python3
"""
Validation script for Flow Assurance Risk Prediction blog code.
Tests all functions to ensure they run without errors.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def generate_pipeline_telemetry(n_segments=4000, seed=77):
    """Generate realistic pipeline segment telemetry data."""
    rng = np.random.default_rng(seed)
    
    crude_types = ['light_sweet', 'medium', 'heavy_waxy']
    crude_probs = [0.5, 0.35, 0.15]
    crude = rng.choice(crude_types, n_segments, p=crude_probs)
    
    wat_celsius = np.where(
        crude == 'heavy_waxy',
        rng.normal(35, 2, n_segments),
        rng.normal(27, 2, n_segments)
    )
    
    temp_in_celsius = rng.normal(32, 4, n_segments)
    temp_out_celsius = temp_in_celsius - rng.normal(3, 2, n_segments).clip(0.5, 10)
    
    pressure_bar = rng.normal(55, 6, n_segments)
    flow_ksm3h = rng.normal(2.0, 0.6, n_segments).clip(0.3, 4.0)
    shear_proxy = 0.3 * flow_ksm3h / (pressure_bar / 50)
    inhibitor_active = rng.choice([0, 1], n_segments, p=[0.7, 0.3])
    water_cut = rng.beta(2, 10, n_segments)
    
    avg_temp = 0.5 * (temp_in_celsius + temp_out_celsius)
    thermal_margin = wat_celsius - avg_temp
    
    base_logit = (
        0.8 * (thermal_margin > 0) +
        0.4 * (thermal_margin > 2) +
        0.3 * (water_cut > 0.2) +
        0.3 * (crude == 'heavy_waxy') -
        0.4 * inhibitor_active -
        0.3 * (shear_proxy > 0.9)
    )
    
    risk_probability = 1 / (1 + np.exp(-(base_logit - 0.4)))
    risk_observed = (rng.random(n_segments) < risk_probability).astype(int)
    
    telemetry = pd.DataFrame({
        'crude_type': crude,
        'wat_celsius': wat_celsius,
        'temp_in_celsius': temp_in_celsius,
        'temp_out_celsius': temp_out_celsius,
        'pressure_bar': pressure_bar,
        'flow_ksm3h': flow_ksm3h,
        'shear_proxy': shear_proxy,
        'inhibitor_active': inhibitor_active,
        'water_cut': water_cut,
        'risk_observed': risk_observed
    })
    
    return telemetry

def train_risk_classifier(telemetry_data, test_size=0.25, random_state=42):
    """Train Random Forest classifier for flow assurance risk prediction."""
    X = telemetry_data.drop(columns=['risk_observed'])
    y = telemetry_data['risk_observed']
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = ['crude_type']
    
    from sklearn.preprocessing import OneHotEncoder
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numeric_features),
            ('categorical', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=3,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'roc_auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

def prioritize_high_risk_segments(model_results, telemetry_data, top_n=20):
    """Rank pipeline segments by risk-adjusted criticality."""
    test_segments = model_results['X_test'].copy()
    test_segments['risk_probability'] = model_results['y_proba']
    test_segments['actual_risk'] = model_results['y_test'].values
    
    test_segments['criticality_score'] = (
        test_segments['pressure_bar'] * 0.6 +
        test_segments['flow_ksm3h'] * 10.0
    )
    
    test_segments['risk_rank_score'] = (
        test_segments['risk_probability'] * 
        (1 + 0.003 * test_segments['criticality_score'])
    )
    
    test_segments['thermal_margin'] = (
        test_segments['wat_celsius'] - 
        0.5 * (test_segments['temp_in_celsius'] + test_segments['temp_out_celsius'])
    )
    
    test_segments['risk_category'] = pd.cut(
        test_segments['risk_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    
    action_map = {
        'LOW': 'Continue monitoring',
        'MEDIUM': 'Increase inspection frequency',
        'HIGH': 'Priority intervention - consider pigging or inhibitor boost'
    }
    test_segments['recommended_action'] = test_segments['risk_category'].map(action_map)
    
    top_segments = test_segments.sort_values('risk_rank_score', ascending=False).head(top_n)
    
    report_columns = [
        'risk_probability', 'risk_rank_score', 'risk_category',
        'crude_type', 'wat_celsius', 'thermal_margin',
        'temp_in_celsius', 'temp_out_celsius',
        'pressure_bar', 'flow_ksm3h',
        'inhibitor_active', 'water_cut',
        'recommended_action'
    ]
    
    return top_segments[report_columns]

def analyze_feature_importance(model_results, feature_names):
    """Extract and analyze feature importance from Random Forest model."""
    rf_model = model_results['model'].named_steps['classifier']
    importances = rf_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    
    return importance_df

def main():
    """Run validation tests."""
    logger.info("FLOW ASSURANCE RISK PREDICTION - CODE VALIDATION")
    
    np.random.seed(77)
    
    logger.info("\n1. Testing pipeline telemetry generation...")
    pipeline_data = generate_pipeline_telemetry(n_segments=4000)
    logger.info(f"   ✓ Generated {len(pipeline_data)} pipeline segments")
    logger.info(f"   ✓ Risk rate: {pipeline_data['risk_observed'].mean():.1%}")
    logger.info(f"   ✓ Crude types: {pipeline_data['crude_type'].nunique()}")
    
    logger.info("\n2. Testing risk classifier training...")
    results = train_risk_classifier(pipeline_data)
    logger.info(f"   ✓ ROC AUC Score: {results['roc_auc']:.3f}")
    logger.info(f"   ✓ Sensitivity: {results['sensitivity']:.1%}")
    logger.info(f"   ✓ Specificity: {results['specificity']:.1%}")
    logger.info(f"   ✓ Precision: {results['precision']:.1%}")
    
    logger.info("\n3. Testing segment prioritization...")
    priority_segments = prioritize_high_risk_segments(results, pipeline_data, top_n=20)
    logger.info(f"   ✓ Top segments identified: {len(priority_segments)}")
    logger.info(f"   ✓ Risk categories: {priority_segments['risk_category'].value_counts().to_dict()}")
    logger.info(f"   ✓ Top risk score: {priority_segments['risk_rank_score'].iloc[0]:.3f}")
    
    logger.info("\n4. Testing feature importance analysis...")
    # Get actual feature names after preprocessing
    numeric_cols = results['X_test'].select_dtypes(include=[np.number]).columns.tolist()
    # OneHotEncoder creates features for each category (minus first which is dropped)
    categorical_cols = ['crude_type_light_sweet', 'crude_type_medium']  # heavy_waxy is dropped
    feature_names = numeric_cols + categorical_cols
    
    feature_importance = analyze_feature_importance(results, feature_names)
    logger.info(f"   ✓ Features analyzed: {len(feature_importance)}")
    logger.info(f"   ✓ Top feature: {feature_importance.iloc[0]['feature']}")
    logger.info(f"   ✓ Top 3 explain: {feature_importance.head(3)['cumulative_importance'].iloc[-1]:.0%}")
    
    logger.info("=== ALL TESTS PASSED! ✓ ===")

if __name__ == "__main__":
    main()

