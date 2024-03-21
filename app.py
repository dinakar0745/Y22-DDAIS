import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
data = pd.read_csv('train.csv')
print("Dataset loaded successfully!")

features = data.drop(['id', 'Product ID', 'Type', 'Machine failure'], axis=1)
target = data['Machine failure']

print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("Dataset splitted successfully!")

print("Creating the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

for epoch in range(10):
    print(f"Training epoch {epoch + 1}...")
    model.fit(X_train, y_train)
    print(f"Epoch {epoch + 1} trained successfully!")

    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the model after epoch {epoch + 1}: {accuracy}")

joblib.dump(model, 'model.h5')
print("Model saved successfully!")
