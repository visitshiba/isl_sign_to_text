import json
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

DATA_FILE = "data/data.json"
MODEL_FILE = "gesture_classifier.pkl"

# Load dataset
with open(DATA_FILE, "r") as f:
    dataset = json.load(f)

X = np.array(dataset["landmarks"])
y = np.array(dataset["labels"])

# Encode labels (strings) to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save model and label map (label encoder classes)
label_map = {label: idx for idx, label in enumerate(le.classes_)}
joblib.dump((model, label_map), MODEL_FILE)
print(f"Model and label map saved to {MODEL_FILE}")
