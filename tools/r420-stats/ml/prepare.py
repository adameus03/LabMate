import numpy as np
import pandas as pd
import os

# ============================================================================
# MAIN FUNCTION: Load CSV files and create .npy datasets
# ============================================================================

def create_datasets_from_csv():
    """
    Load RFID data from CSV files and create .npy training/validation datasets.
    
    Expected CSV format:
    - Each CSV file should have 208 columns (one per feature)
    - Each row represents one measurement sample
    - No header row expected (set has_header=True if your CSVs have headers)
    """
    
    # Configuration
    csv_files = {
        'train': {
            'piano': '../piano_1.csv',           # Dataset 1: Piano location
            'table1': '../stol_l_1.csv',         # Dataset 2: Table segment 1
            'table2': '../stol_r_1.csv'          # Dataset 3: Table segment 2
        },
        'val': {
            'piano': '../piano_2.csv',             # Validation 1: Piano
            'table1': '../stol_l_2.csv',           # Validation 2: Table segment 1
            'table2': '../stol_r_2.csv'            # Validation 3: Table segment 2
        }
    }
    
    has_header = False  # Set to True if your CSV files have a header row
    
    # Process training datasets
    print("=" * 60)
    print("CREATING TRAINING DATASETS")
    print("=" * 60)
    
    for idx, (location_name, csv_file) in enumerate(csv_files['train'].items(), start=1):
        process_csv_file(
            csv_file=csv_file,
            location_name=location_name,
            location_idx=idx - 1,  # 0, 1, 2
            output_prefix=f'dataset{idx}',
            has_header=has_header
        )
    
    # Process validation datasets
    print("\n" + "=" * 60)
    print("CREATING VALIDATION DATASETS")
    print("=" * 60)
    
    for idx, (location_name, csv_file) in enumerate(csv_files['val'].items(), start=1):
        process_csv_file(
            csv_file=csv_file,
            location_name=location_name,
            location_idx=idx - 1,  # 0, 1, 2
            output_prefix=f'val{idx}',
            has_header=has_header
        )
    
    print("\n" + "=" * 60)
    print("✓ ALL DATASETS CREATED SUCCESSFULLY!")
    print("=" * 60)
    verify_all_datasets()


# ============================================================================
# HELPER FUNCTION: Process a single CSV file
# ============================================================================

def process_csv_file(csv_file, location_name, location_idx, output_prefix, has_header):
    """
    Load a single CSV file and create corresponding .npy files.
    
    Args:
        csv_file: Path to the CSV file
        location_name: Name of the location (for display)
        location_idx: Index of the location (0=piano, 1=table1, 2=table2)
        output_prefix: Prefix for output files (e.g., 'dataset1' or 'val1')
        has_header: Whether the CSV has a header row
    """
    
    print(f"\nProcessing {location_name} from '{csv_file}'...")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"  ⚠ WARNING: File '{csv_file}' not found. Skipping...")
        return
    
    try:
        # Load CSV file
        if has_header:
            df = pd.read_csv(csv_file, header=0)
        else:
            df = pd.read_csv(csv_file, header=None)
        
        # Convert to numpy array
        inputs = df.values.astype(np.float32)
        num_samples = inputs.shape[0]
        
        # Validate shape
        if inputs.shape[1] != 208:
            print(f"  ✗ ERROR: Expected 208 features, but got {inputs.shape[1]}")
            print(f"    Please check your CSV file format.")
            return
        
        # Create one-hot encoded outputs
        # Location 0 (piano) = [1, 0, 0]
        # Location 1 (table1) = [0, 1, 0]
        # Location 2 (table2) = [0, 0, 1]
        outputs = np.zeros((num_samples, 3), dtype=np.float32)
        outputs[:, location_idx] = 1.0
        
        # Save to .npy files
        input_filename = f'{output_prefix}_inputs.npy'
        output_filename = f'{output_prefix}_outputs.npy'
        
        np.save(input_filename, inputs)
        np.save(output_filename, outputs)
        
        print(f"  ✓ Loaded {num_samples} samples")
        print(f"  ✓ Saved to '{input_filename}' and '{output_filename}'")
        
    except Exception as e:
        print(f"  ✗ ERROR processing file: {e}")


# ============================================================================
# VERIFICATION FUNCTION
# ============================================================================

def verify_all_datasets():
    """
    Load and verify all created .npy files.
    """
    
    print("\nVerifying created datasets...")
    print("-" * 60)
    
    dataset_files = [
        ('dataset1', 'Piano (train)'),
        ('dataset2', 'Table segment 1 (train)'),
        ('dataset3', 'Table segment 2 (train)'),
        ('val1', 'Piano (val)'),
        ('val2', 'Table segment 1 (val)'),
        ('val3', 'Table segment 2 (val)')
    ]
    
    all_valid = True
    
    for prefix, name in dataset_files:
        input_file = f'{prefix}_inputs.npy'
        output_file = f'{prefix}_outputs.npy'
        
        if not os.path.exists(input_file) or not os.path.exists(output_file):
            print(f"  ⚠ {name}: Files not found (skipped during creation)")
            continue
        
        try:
            inputs = np.load(input_file)
            outputs = np.load(output_file)
            
            # Verify shapes
            assert inputs.shape[1] == 208, f"Inputs should have 208 features, got {inputs.shape[1]}"
            assert outputs.shape[1] == 3, f"Outputs should have 3 classes, got {outputs.shape[1]}"
            assert inputs.shape[0] == outputs.shape[0], "Inputs and outputs should have same number of samples"
            assert np.allclose(outputs.sum(axis=1), 1.0), "Each output should sum to 1"
            
            print(f"  ✓ {name}: {inputs.shape[0]} samples, shape validated")
            
        except Exception as e:
            print(f"  ✗ {name}: Validation failed - {e}")
            all_valid = False
    
    print("-" * 60)
    if all_valid:
        print("✓ All datasets are valid and ready for training!")
    else:
        print("⚠ Some datasets had validation errors.")


# ============================================================================
# UTILITY: Inspect a CSV file before processing
# ============================================================================

def inspect_csv_file(csv_file):
    """
    Quick inspection of a CSV file to check format.
    """
    print(f"\nInspecting '{csv_file}'...")
    
    if not os.path.exists(csv_file):
        print(f"  ✗ File not found!")
        return
    
    # Try to read first few rows
    try:
        df = pd.read_csv(csv_file, header=None, nrows=5)
        print(f"  Number of columns: {df.shape[1]}")
        print(f"  First few rows:")
        print(df.head())
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Optional: Inspect your CSV files first
    # Uncomment these lines to check your CSV format:
    # inspect_csv_file('piano_train.csv')
    # inspect_csv_file('table1_train.csv')
    
    # Create all datasets from CSV files
    create_datasets_from_csv()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Verify the shapes look correct above")
    print("2. Use these .npy files in your training script")
    print("3. Files created:")
    print("   - dataset1_inputs.npy, dataset1_outputs.npy (Piano training)")
    print("   - dataset2_inputs.npy, dataset2_outputs.npy (Table1 training)")
    print("   - dataset3_inputs.npy, dataset3_outputs.npy (Table2 training)")
    print("   - val1_inputs.npy, val1_outputs.npy (Piano validation)")
    print("   - val2_inputs.npy, val2_outputs.npy (Table1 validation)")
    print("   - val3_inputs.npy, val3_outputs.npy (Table2 validation)")