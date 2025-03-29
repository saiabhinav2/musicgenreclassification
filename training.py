import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# Load dataset
df = pd.read_csv('features_3_sec.csv')

# Encode labels
class_encod = df.iloc[:, -1]  # Assuming last column is the label
converter = LabelEncoder()
y = converter.fit_transform(class_encod)

# Drop label column to get feature matrix (X)
df = df.drop(labels="filename", axis=1)  # Adjust the column names accordingly
X = df.iloc[:, :-1]  # Exclude label column

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(np.array(X, dtype=float))

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Build the model
model = Sequential([
    Flatten(input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    
    Dense(512, activation='relu'),
    Dropout(0.2),
    
    Dense(256, activation='relu'),
    Dropout(0.2),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    Dense(10, activation='softmax'),  # 10 classes, adjust as needed
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=256)

# Save the trained model
model.save('models/trained_model.h5')
