import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_ev_data(filepath):
    """Load and clean electric vehicle data"""
    try:
        # 1. Load and inspect data
        ev_data = pd.read_csv(filepath)
        print("Data Overview:")
        print(ev_data.head())
        print("\nData Info:")
        print(ev_data.info())

        # 2. Clean data
        # Handle missing values and duplicates
        ev_data = ev_data.drop_duplicates()
        
        # Verify Electric Range column exists and is numeric
        if 'Electric Range' not in ev_data.columns:
            raise ValueError("'Electric Range' column not found in dataset")
            
        if not pd.api.types.is_numeric_dtype(ev_data['Electric Range']):
            ev_data['Electric Range'] = pd.to_numeric(ev_data['Electric Range'], errors='coerce')
        
        median_range = ev_data['Electric Range'].median()
        ev_data['Electric Range'] = ev_data['Electric Range'].fillna(median_range)

        # Clean up text data
        if 'Make' in ev_data.columns:
            ev_data['Make'] = ev_data['Make'].str.upper().str.strip()
        else:
            print("Warning: 'Make' column not found")

        # 3. Normalize electric range
        scaler = MinMaxScaler()
        ev_data['Normalized Range'] = scaler.fit_transform(ev_data[['Electric Range']])

        # Verify normalization
        if not ev_data['Normalized Range'].between(0, 1).all():
            print("Warning: Normalization produced values outside [0,1] range")

        # 4. Select important columns
        keep_cols = ['Make', 'Model', 'Model Year', 'Electric Range', 'Normalized Range']
        available_cols = [col for col in keep_cols if col in ev_data.columns]
        ev_data = ev_data[available_cols].copy()  # Explicit copy to optimize memory

        # 5. Create range categories
        range_bins = [0, 50, 100, 150, 200, 300, float('inf')]
        range_labels = ['0-50', '51-100', '101-150', '151-200', '201-300', '300+']
        ev_data['Range Group'] = pd.cut(ev_data['Electric Range'], 
                                      bins=range_bins, 
                                      labels=range_labels)

        # Show results
        print("\nNormalized Range Stats:")
        print(ev_data[['Electric Range', 'Normalized Range']].describe())
        print("\nVehicle Range Distribution:")
        print(ev_data['Range Group'].value_counts())
        
        return ev_data

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Run the function with your file path
cleaned_data = load_and_clean_ev_data("Electric_Vehicle_Population_Data.csv")

if cleaned_data is not None:
    print("\nData cleaning and processing completed successfully!")
    # You can now use cleaned_data for further analysis