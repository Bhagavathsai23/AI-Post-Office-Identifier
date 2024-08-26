import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Simulate a dataset
data = {
    'Address': [
        '123 Main St, Area A', '456 Park Ave, Area B', '789 Oak St, Area C', 
        '101 Maple St, Area A', '202 Pine St, Area B', '303 Elm St, Area C'
    ],
    'PinCode': [110001, 110002, 110003, 110001, 110002, 110003],
    'DeliveryPostOffice': ['PostOffice_A', 'PostOffice_B', 'PostOffice_C', 
                           'PostOffice_A', 'PostOffice_B', 'PostOffice_C']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Preprocess the data
label_encoder = LabelEncoder()
df['Address_Encoded'] = label_encoder.fit_transform(df['Address'])
df['PostOffice_Encoded'] = label_encoder.fit_transform(df['DeliveryPostOffice'])

# Split the data into features and target
X = df[['PinCode', 'Address_Encoded']]
y = df['PostOffice_Encoded']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
