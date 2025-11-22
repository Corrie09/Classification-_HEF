#EDA done op 22/11/2025 23:00 op de 2de preprocessed dataset (van 22/11) om daarna de 3de preprocessed dataset (van 22/11) te maken met verbeteringen uit deze EDA
# # =============================================================================
# PREPROCESSING DIAGNOSTICS - Find Improvement Opportunities
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load current preprocessed data
X = pd.read_pickle('data/processed/X_train_processed.pkl')
y = pd.read_pickle('data/processed/y_train.pkl')

print("="*70)
print("PREPROCESSING DIAGNOSTICS")
print("="*70)

# =============================================================================
# 1. FEATURE DISTRIBUTIONS - Check for Skewness
# =============================================================================
print("\n" + "="*70)
print("1. HIGHLY SKEWED FEATURES (may need transformation)")
print("="*70)

# Get numeric features only
numeric_cols = X.select_dtypes(include=[np.number]).columns

skewness = X[numeric_cols].skew().sort_values(ascending=False)
highly_skewed = skewness[abs(skewness) > 2]

if len(highly_skewed) > 0:
    print(f"\nFound {len(highly_skewed)} highly skewed features (|skew| > 2):")
    for feat, skew in highly_skewed.items():
        print(f"  {feat}: {skew:.2f}")
else:
    print("No highly skewed features found")

# =============================================================================
# 2. FEATURES WITH EXTREME OUTLIERS
# =============================================================================
print("\n" + "="*70)
print("2. FEATURES WITH EXTREME OUTLIERS")
print("="*70)

outlier_analysis = []

for col in numeric_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
    upper_bound = Q3 + 3 * IQR
    
    outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
    outlier_pct = outliers / len(X) * 100
    
    if outlier_pct > 1:  # More than 1% outliers
        outlier_analysis.append({
            'feature': col,
            'outlier_count': outliers,
            'outlier_pct': outlier_pct,
            'min': X[col].min(),
            'max': X[col].max(),
            'median': X[col].median()
        })

if outlier_analysis:
    outlier_df = pd.DataFrame(outlier_analysis).sort_values('outlier_pct', ascending=False)
    print(f"\nFeatures with >1% extreme outliers:")
    print(outlier_df.to_string(index=False))
else:
    print("No features with significant outliers")

# =============================================================================
# 3. ZERO/NEAR-ZERO VARIANCE FEATURES
# =============================================================================
print("\n" + "="*70)
print("3. LOW VARIANCE FEATURES (may not be useful)")
print("="*70)

# Calculate variance for numeric features
variances = X[numeric_cols].var().sort_values()
low_variance = variances[variances < 0.01]

if len(low_variance) > 0:
    print(f"\nFound {len(low_variance)} features with variance < 0.01:")
    for feat, var in low_variance.items():
        unique_vals = X[feat].nunique()
        print(f"  {feat}: variance={var:.6f}, unique_values={unique_vals}")
else:
    print("No low variance features")

# =============================================================================
# 4. HIGHLY CORRELATED FEATURES (Redundant Information)
# =============================================================================
print("\n" + "="*70)
print("4. HIGHLY CORRELATED FEATURE PAIRS (may be redundant)")
print("="*70)

# Calculate correlation matrix
corr_matrix = X[numeric_cols].corr().abs()

# Get upper triangle (avoid duplicates)
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Find highly correlated pairs (>0.9)
high_corr_pairs = []
for column in upper_triangle.columns:
    high_corr = upper_triangle[column][upper_triangle[column] > 0.9]
    for index in high_corr.index:
        high_corr_pairs.append({
            'feature_1': column,
            'feature_2': index,
            'correlation': corr_matrix.loc[index, column]
        })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
    print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (corr > 0.9):")
    print(high_corr_df.to_string(index=False))
else:
    print("No highly correlated feature pairs found")

# =============================================================================
# 5. FEATURE IMPORTANCE FROM SIMPLE MODEL
# =============================================================================
print("\n" + "="*70)
print("5. FEATURE IMPORTANCE (from quick Random Forest)")
print("="*70)

from sklearn.ensemble import RandomForestClassifier

# Quick RF to get feature importance
rf_quick = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_quick.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_quick.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importance.head(20).to_string(index=False))

print("\nBottom 10 least important features:")
print(feature_importance.tail(10).to_string(index=False))

# How many features have near-zero importance?
zero_importance = (feature_importance['importance'] < 0.001).sum()
print(f"\nFeatures with near-zero importance (<0.001): {zero_importance}")

# =============================================================================
# 6. TARGET CORRELATION ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("6. FEATURE CORRELATION WITH TARGET")
print("="*70)

# Calculate correlation with target
target_corr = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)

print("\nTop 15 features most correlated with mortality:")
print(target_corr.head(15).to_string())

print("\nBottom 10 features least correlated with mortality:")
print(target_corr.tail(10).to_string())

weak_correlation = target_corr[target_corr < 0.01]
print(f"\nFeatures with very weak correlation (<0.01): {len(weak_correlation)}")

# =============================================================================
# 7. MISSING VALUE PATTERNS (in original data)
# =============================================================================
print("\n" + "="*70)
print("7. MISSING VALUE ANALYSIS (reload original to check)")
print("="*70)

train_original = pd.read_csv('data/mimic_train_HEF.csv')

# Check missing values before imputation
missing_before = train_original.isnull().sum()
missing_before = missing_before[missing_before > 0].sort_values(ascending=False)

if len(missing_before) > 0:
    missing_pct = (missing_before / len(train_original) * 100).round(2)
    missing_df = pd.DataFrame({
        'feature': missing_before.index,
        'missing_count': missing_before.values,
        'missing_pct': missing_pct.values
    })
    print("\nFeatures with missing values in original data:")
    print(missing_df.to_string(index=False))
else:
    print("No missing values in original data")

# =============================================================================
# 8. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*70)
print("8. SUMMARY")
print("="*70)

print(f"\nTotal features: {X.shape[1]}")
print(f"Total samples: {X.shape[0]}")
print(f"Target prevalence: {y.mean():.3f} ({y.sum()} deaths)")

print(f"\nFeature types:")
print(f"  Continuous numeric: ~{len([col for col in numeric_cols if X[col].nunique() > 20])}")
print(f"  Binary: ~{len([col for col in X.columns if X[col].nunique() == 2])}")
print(f"  Categorical (one-hot): ~{X.shape[1] - len(numeric_cols)}")

print("\n" + "="*70)
print("DIAGNOSTICS COMPLETE")
print("="*70)