import numpy as np

# ============================================================================
# METHOD 1: If you're collecting data from your RFID system
# ============================================================================

def create_from_rfid_measurements():
    """
    Example: Collecting real RFID data and saving it
    """
    
    # Collect measurements for Piano location
    piano_measurements = []
    
    # Assume you have a function that reads from your RFID system
    # that returns data shaped (13, 2, 4, 2)
    for i in range(1000):  # Collect 1000 samples
        measurement = read_from_rfid_system()  # Your function here
        # measurement shape: (13 tags, 2 antennas, 4 features, 2 readings)
        
        # Flatten to 208 dimensions
        flattened = measurement.reshape(208)
        piano_measurements.append(flattened)
    
    # Convert to numpy array
    piano_inputs = np.array(piano_measurements)  # Shape: (1000, 208)
    
    # Create outputs (one-hot encoded labels)
    # Piano = [1, 0, 0]
    piano_outputs = np.zeros((1000, 3))
    piano_outputs[:, 0] = 1.0  # Set first column to 1
    
    # Save to files
    np.save('dataset1_inputs.npy', piano_inputs)
    np.save('dataset1_outputs.npy', piano_outputs)
    
    print("Piano dataset saved!")
    print(f"Inputs shape: {piano_inputs.shape}")
    print(f"Outputs shape: {piano_outputs.shape}")


# ============================================================================
# METHOD 2: If you have data in CSV files
# ============================================================================

def create_from_csv():
    """
    Example: Loading from CSV and converting to .npy
    """
    import pandas as pd
    
    # Load your CSV (adjust column names as needed)
    df = pd.read_csv('piano_measurements.csv')
    
    # Assume CSV has 208 columns for features and 1 column for location
    inputs = df.iloc[:, :208].values  # First 208 columns
    
    # Create one-hot encoded outputs
    # If location is 0 (piano), 1 (table1), or 2 (table2)
    num_samples = len(df)
    outputs = np.zeros((num_samples, 3))
    
    location_labels = df['location'].values  # Assuming a 'location' column
    for i, label in enumerate(location_labels):
        outputs[i, label] = 1.0
    
    # Save
    np.save('dataset1_inputs.npy', inputs)
    np.save('dataset1_outputs.npy', outputs)


# ============================================================================
# METHOD 3: Creating example/synthetic data for testing
# ============================================================================

def create_synthetic_data():
    """
    Create synthetic data to test your training pipeline
    """
    num_samples_train = 100
    num_samples_val = 100
    
    # Training Dataset 1: Piano (location 0)
    piano_inputs_train = np.random.randn(num_samples_train, 208).astype(np.float32)
    piano_outputs_train = np.zeros((num_samples_train, 3), dtype=np.float32)
    piano_outputs_train[:, 0] = 1.0  # [1, 0, 0]
    
    np.save('dataset1_inputs.npy', piano_inputs_train)
    np.save('dataset1_outputs.npy', piano_outputs_train)
    
    # Training Dataset 2: Table segment 1 (location 1)
    table1_inputs_train = np.random.randn(num_samples_train, 208).astype(np.float32)
    table1_outputs_train = np.zeros((num_samples_train, 3), dtype=np.float32)
    table1_outputs_train[:, 1] = 1.0  # [0, 1, 0]
    
    np.save('dataset2_inputs.npy', table1_inputs_train)
    np.save('dataset2_outputs.npy', table1_outputs_train)
    
    # Training Dataset 3: Table segment 2 (location 2)
    table2_inputs_train = np.random.randn(num_samples_train, 208).astype(np.float32)
    table2_outputs_train = np.zeros((num_samples_train, 3), dtype=np.float32)
    table2_outputs_train[:, 2] = 1.0  # [0, 0, 1]
    
    np.save('dataset3_inputs.npy', table2_inputs_train)
    np.save('dataset3_outputs.npy', table2_outputs_train)

    # Val Dataset 1: Piano (location 0)
    piano_inputs_val = np.random.randn(num_samples_val, 208).astype(np.float32)
    piano_outputs_val = np.zeros((num_samples_val, 3), dtype=np.float32)
    piano_outputs_val[:, 0] = 1.0  # [1, 0, 0]
    
    np.save('val1_inputs.npy', piano_inputs_val)
    np.save('val1_outputs.npy', piano_outputs_val)
    
    # Val Dataset 2: Table segment 1 (location 1)
    table1_inputs_val = np.random.randn(num_samples_val, 208).astype(np.float32)
    table1_outputs_val = np.zeros((num_samples_val, 3), dtype=np.float32)
    table1_outputs_val[:, 1] = 1.0  # [0, 1, 0]
    
    np.save('val2_inputs.npy', table1_inputs_val)
    np.save('val2_outputs.npy', table1_outputs_val)
    
    # Training Dataset 3: Table segment 2 (location 2)
    table2_inputs_val = np.random.randn(num_samples_val, 208).astype(np.float32)
    table2_outputs_val = np.zeros((num_samples_val, 3), dtype=np.float32)
    table2_outputs_val[:, 2] = 1.0  # [0, 0, 1]
    
    np.save('val3_inputs.npy', table2_inputs_val)
    np.save('val3_outputs.npy', table2_outputs_val)
    
    print("Synthetic datasets created!")


# ============================================================================
# METHOD 4: If your data is in a different format (JSON, text files, etc.)
# ============================================================================

def create_from_text_files():
    """
    Example: Reading from text files line by line
    """
    inputs = []
    outputs = []
    
    with open('piano_data.txt', 'r') as f:
        for line in f:
            # Assume each line has 208 comma-separated values
            values = [float(x) for x in line.strip().split(',')]
            inputs.append(values)
            outputs.append([1.0, 0.0, 0.0])  # Piano label
    
    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    
    np.save('dataset1_inputs.npy', inputs)
    np.save('dataset1_outputs.npy', outputs)


# ============================================================================
# LOADING AND VERIFYING YOUR DATA
# ============================================================================

def load_and_verify():
    """
    How to load and check your .npy files
    """
    # Load the files
    inputs = np.load('dataset1_inputs.npy')
    outputs = np.load('dataset1_outputs.npy')
    
    print("Data loaded successfully!")
    print(f"Inputs shape: {inputs.shape}")  # Should be (num_samples, 208)
    print(f"Outputs shape: {outputs.shape}")  # Should be (num_samples, 3)
    print(f"\nFirst input sample:\n{inputs[0]}")
    print(f"\nFirst output sample:\n{outputs[0]}")
    
    # Verify the data looks correct
    assert inputs.shape[1] == 208, "Inputs should have 208 features!"
    assert outputs.shape[1] == 3, "Outputs should have 3 classes!"
    assert np.allclose(outputs.sum(axis=1), 1.0), "Each output should sum to 1!"


# ============================================================================
# EXAMPLE: Complete workflow for your RFID system
# ============================================================================

def complete_example():
    """
    Step-by-step example for your RFID RTLS system
    """
    
    # Step 1: Collect data for each location
    locations = ['piano', 'table_segment1', 'table_segment2']
    
    for location_idx, location_name in enumerate(locations):
        print(f"\nCollecting data for {location_name}...")
        
        measurements = []
        
        # Collect N samples for this location
        for sample_num in range(500):  # 500 samples per location
            # Your RFID reading function
            # Returns shape (13, 2, 4, 2)
            raw_measurement = get_rfid_reading()  # YOU IMPLEMENT THIS
            
            # Flatten to (208,)
            flattened = raw_measurement.flatten()
            measurements.append(flattened)
        
        # Convert to numpy
        inputs = np.array(measurements, dtype=np.float32)
        
        # Create one-hot labels
        outputs = np.zeros((len(measurements), 3), dtype=np.float32)
        outputs[:, location_idx] = 1.0
        
        # Save
        np.save(f'dataset{location_idx + 1}_inputs.npy', inputs)
        np.save(f'dataset{location_idx + 1}_outputs.npy', outputs)
        
        print(f"Saved {len(measurements)} samples for {location_name}")


# ============================================================================
# RUN ONE OF THESE
# ============================================================================

if __name__ == "__main__":
    # Choose one:
    
    # For testing your training code:
    create_synthetic_data()
    
    # Then verify it worked:
    load_and_verify()
    
    print("\n✓ Files created successfully!")
    print("You can now use these in your training script.")