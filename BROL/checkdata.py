# =============================================================================
# EXPLORE EXTRA DATA SOURCES
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path("data/")

print("="*70)
print("EXPLORING AVAILABLE DATA FILES")
print("="*70)

# Check what files exist
files = list(data_path.glob("*.csv"))
print("\nAvailable CSV files:")
for f in files:
    size_mb = f.stat().st_size / (1024*1024)
    print(f"  {f.name}: {size_mb:.2f} MB")

# Load main datasets
train = pd.read_csv(data_path / "mimic_train_HEF.csv", low_memory=False)
test = pd.read_csv(data_path / "mimic_test_HEF.csv", low_memory=False)

print("\n" + "="*70)
print("MAIN DATASET INFO")
print("="*70)

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")

print(f"\nID columns in train:")
id_cols = ['subject_id', 'hadm_id', 'icustay_id']
for col in id_cols:
    if col in train.columns:
        n_unique = train[col].nunique()
        print(f"  {col}: {n_unique:,} unique values")

# Check if extra_data folder exists
extra_data_path = data_path / "extra_data"
if extra_data_path.exists():
    print("\n" + "="*70)
    print("EXTRA DATA FILES")
    print("="*70)
    
    extra_files = list(extra_data_path.glob("*.csv"))
    for f in extra_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name}: {size_mb:.2f} MB")
        
        # Quick peek at diagnoses file if it exists
        if 'diagnoses' in f.name.lower():
            print(f"\n    Loading {f.name} for inspection...")
            diagnoses = pd.read_csv(f, nrows=1000)
            print(f"    Shape: {diagnoses.shape}")
            print(f"    Columns: {diagnoses.columns.tolist()}")
            print(f"    Sample:")
            print(diagnoses.head(3))
else:
    print("\n⚠️ No 'extra_data' folder found")
    print("   Check if MIMIC_diagnoses.csv is in the main data folder")
    
    # Check main folder for diagnoses
    diag_files = list(data_path.glob("*diagnoses*.csv"))
    if diag_files:
        print(f"\n   Found in main folder: {[f.name for f in diag_files]}")

print("\n" + "="*70)
print("PATIENT VISIT HISTORY ANALYSIS")
print("="*70)

# Analyze hospital visits per patient
if 'subject_id' in train.columns and 'hadm_id' in train.columns:
    
    print("\nPatients and admissions:")
    print(f"  Unique patients (subject_id): {train['subject_id'].nunique():,}")
    print(f"  Unique hospital admissions (hadm_id): {train['hadm_id'].nunique():,}")
    print(f"  Unique ICU stays (icustay_id): {train['icustay_id'].nunique():,}")
    
    # Count visits per patient
    visits_per_patient = train.groupby('subject_id').size()
    print(f"\nICU stays per patient:")
    print(f"  Mean: {visits_per_patient.mean():.2f}")
    print(f"  Median: {visits_per_patient.median():.0f}")
    print(f"  Max: {visits_per_patient.max():.0f}")
    
    # Patients with multiple visits
    repeat_patients = (visits_per_patient > 1).sum()
    print(f"  Patients with >1 ICU stay: {repeat_patients:,} ({repeat_patients/len(visits_per_patient)*100:.1f}%)")
    
    # Hospital admissions per patient
    admissions_per_patient = train.groupby('subject_id')['hadm_id'].nunique()
    print(f"\nHospital admissions per patient:")
    print(f"  Mean: {admissions_per_patient.mean():.2f}")
    print(f"  Median: {admissions_per_patient.median():.0f}")
    print(f"  Max: {admissions_per_patient.max():.0f}")

print("\n" + "="*70)