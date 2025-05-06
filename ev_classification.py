# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Load dataset
ev_data = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# Data Preprocessing 
# ------------------------------
# 1. Handle missing values in Electric Range (critical for binning)
ev_data['Electric Range'] = ev_data['Electric Range'].fillna(ev_data['Electric Range'].median())

# 2. Binning: Convert 'Electric Range' into 4 classes
bins = [0, 100, 200, 300, float('inf')]
labels = ['Short (0-100)', 'Medium (101-200)', 'Long (201-300)', 'Very Long (300+)']
ev_data['Range_Class'] = pd.cut(ev_data['Electric Range'], bins=bins, labels=labels)

# 3. Feature Engineering
# Handle missing Model Year values first
ev_data['Model Year'] = ev_data['Model Year'].fillna(ev_data['Model Year'].median())
ev_data['Age'] = 2023 - ev_data['Model Year']  # Vehicle age

# 4. One-Hot Encoding for 'Make'
# First handle missing Make values
ev_data['Make'] = ev_data['Make'].fillna('Unknown')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
make_encoded = encoder.fit_transform(ev_data[['Make']])
make_encoded_df = pd.DataFrame(make_encoded, columns=encoder.get_feature_names_out(['Make']))
ev_data = pd.concat([ev_data, make_encoded_df], axis=1)

# 5. Final data cleaning - drop any remaining rows with missing target values
ev_data = ev_data.dropna(subset=['Range_Class'])

# Classification 
# --------------------------
# Define features (X) and target (y)
features = ['Age', 'Electric Range'] + list(make_encoded_df.columns)
X = ev_data[features]
y = ev_data['Range_Class']

# Verify no NaN values remain
print("NaN values in features:", X.isna().sum().sum())
print("NaN values in target:", y.isna().sum())

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTop 10 Important Features:\n", importance.head(10))

# Additional useful outputs
print("\nClass Distribution:")
print(y.value_counts())