#EDA done op 22/11/2025 23:14 op de 3de preprocessed dataset (van 22/11) om daarna de 4de preprocessed dataset (van 22/11) te maken met verbeteringen uit deze EDA
# =============================================================================
# COMPARE: Before vs After Changes
# =============================================================================

import pandas as pd
import numpy as np

# Load the NEW preprocessed data
X_new = pd.read_pickle('data/processed/X_train_processed.pkl')
y = pd.read_pickle('data/processed/y_train.pkl')

print("="*70)
print("WHAT CHANGED? - Comparing Preprocessing Versions")
print("="*70)

# =============================================================================
# 1. FEATURE COUNT
# =============================================================================
print(f"\nFeature count: {X_new.shape[1]}")
print("Expected: ~70 (was 74, dropped 2 redundant)")

# =============================================================================
# 2. CHECK BINARY FEATURES - Did they stay 0/1?
# =============================================================================
print("\n" + "="*70)
print("CHECKING BINARY FEATURES (Should be 0/1)")
print("="*70)

binary_indicators = ['Hypoxemia', 'RespRate_Abnormal', 'Fever', 'Hypothermia', 
                     'Hyperglycemia', 'Hypoglycemia', 'Elderly']

binary_status = []
for feat in binary_indicators:
    if feat in X_new.columns:
        unique_vals = sorted(X_new[feat].unique())
        variance = X_new[feat].var()
        binary_status.append({
            'feature': feat,
            'unique_values': unique_vals,
            'variance': variance,
            'min': X_new[feat].min(),
            'max': X_new[feat].max()
        })

if binary_status:
    print("\nBinary feature status:")
    for item in binary_status:
        print(f"\n{item['feature']}:")
        print(f"  Unique values: {item['unique_values']}")
        print(f"  Variance: {item['variance']:.6f}")
        print(f"  Range: [{item['min']:.4f}, {item['max']:.4f}]")
        
        # Check if it's still binary
        if len(item['unique_values']) == 1:
            print(f"  ⚠️ CONSTANT! All values are {item['unique_values'][0]}")
        elif set(item['unique_values']).issubset({0, 1, 0.0, 1.0}):
            print(f"  ✓ Still binary (0/1)")
        else:
            print(f"  ❌ NOT BINARY ANYMORE - Got scaled!")

# =============================================================================
# 3. CHECK IF REDUNDANT FEATURES WERE DROPPED
# =============================================================================
print("\n" + "="*70)
print("CHECKING DROPPED FEATURES")
print("="*70)

should_be_dropped = ['RespDistress_Score', 'MeanBP_Mean']
still_present = [feat for feat in should_be_dropped if feat in X_new.columns]

if still_present:
    print(f"⚠️ Features that SHOULD be dropped but are still present: {still_present}")
else:
    print(f"✓ Redundant features successfully dropped: {should_be_dropped}")

# =============================================================================
# 4. CHECK SHOCK INDICES - Were they clipped?
# =============================================================================
print("\n" + "="*70)
print("CHECKING SHOCK INDICES")
print("="*70)

shock_features = ['ShockIndex', 'ModifiedShockIndex']
for feat in shock_features:
    if feat in X_new.columns:
        print(f"\n{feat}:")
        print(f"  Min: {X_new[feat].min():.4f}")
        print(f"  Max: {X_new[feat].max():.4f}")
        print(f"  Mean: {X_new[feat].mean():.4f}")
        print(f"  Std: {X_new[feat].std():.4f}")
        
        if X_new[feat].max() > 10:
            print(f"  ❌ NOT CLIPPED - Still has extreme values!")
        else:
            print(f"  ✓ Clipped successfully")

# =============================================================================
# 5. CHECK FOR NaN or INFINITE VALUES
# =============================================================================
print("\n" + "="*70)
print("CHECKING FOR INVALID VALUES")
print("="*70)

nan_count = X_new.isnull().sum().sum()
inf_count = np.isinf(X_new.select_dtypes(include=[np.number])).sum().sum()

print(f"NaN values: {nan_count}")
print(f"Infinite values: {inf_count}")

if nan_count > 0:
    print("\n⚠️ Features with NaN:")
    print(X_new.isnull().sum()[X_new.isnull().sum() > 0])

if inf_count > 0:
    print("\n⚠️ Features with Inf values:")
    inf_cols = X_new.select_dtypes(include=[np.number]).columns[np.isinf(X_new.select_dtypes(include=[np.number])).any()]
    print(inf_cols.tolist())

# =============================================================================
# 6. SUMMARY
# =============================================================================
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

issues_found = []

# Check for constant features
constant_features = [feat for feat in binary_indicators if feat in X_new.columns and X_new[feat].var() == 0]
if constant_features:
    issues_found.append(f"❌ {len(constant_features)} constant binary features (zero variance)")

# Check if redundant features still there
if still_present:
    issues_found.append(f"❌ {len(still_present)} redundant features not dropped")

# Check if shock indices clipped
for feat in shock_features:
    if feat in X_new.columns and X_new[feat].max() > 10:
        issues_found.append(f"❌ {feat} not clipped (max={X_new[feat].max():.1f})")

# Check for invalid values
if nan_count > 0:
    issues_found.append(f"❌ {nan_count} NaN values present")
if inf_count > 0:
    issues_found.append(f"❌ {inf_count} Infinite values present")

if issues_found:
    print("\n🚨 ISSUES FOUND:")
    for issue in issues_found:
        print(f"  {issue}")
    print("\nThese issues likely caused the performance drop!")
else:
    print("\n✓ No obvious issues found in preprocessing")
    print("Performance drop might be due to:")
    print("  - Removed features were actually useful")
    print("  - Log transformation hurt performance")
    print("  - Random variation in Kaggle scoring")

print("\n" + "="*70)