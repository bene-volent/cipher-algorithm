import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from feature_engineering import generate_features
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Function to generate dataset from CSV file
def create_dataset(filename: str) -> pd.DataFrame:
    """Load dataset from a CSV file and generate features."""
    df = pd.read_csv(filename)
    
    data = []
    for _, row in df.iterrows():
        hex_input = row['Cipher']
        algo = row['Algorithm']
        features = generate_features(hex_input)
        data.append(np.append(features, algo))  # Append features and the label
    
    features = [d[:-1] for d in data]  # Features
    labels = [d[-1] for d in data]     # Labels
    dataset = pd.DataFrame(features, columns=['Byte_Freq_' + str(i) for i in range(256)] + ['Entropy', 'Block_Size', 'Byte_Mean', 'Byte_Std'])
    dataset['Algorithm'] = labels
    return dataset

# Function to train and save models
def train_and_save_models(hex_samples="cryptographic_algorithms_dataset.csv"):
    """Train multiple models and save them to disk."""
    # Generate dataset
    dataset = create_dataset(hex_samples)
    X = dataset.iloc[:, :-1].astype(float)  # Features
    y = dataset['Algorithm']  # Labels (Algorithms)
    y_encoded = label_encoder.fit_transform(y)  # Encode labels
    
    # Save the label encoder
    
    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Define and train models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear'),
        "KNN": KNeighborsClassifier(n_neighbors=4)
    }
    
    for model_name, model in models.items():
        print(f"Training {model_name} model...")
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Model Accuracy: {accuracy * 100:.2f}%")
        
        # Save the trained model
        joblib.dump(model, f"{model_name}_model.pkl")
        joblib.dump(label_encoder, f"label_encoder-{model_name}.pkl")
        
        

if __name__ == "__main__":
    train_and_save_models(hex_samples='cryptographic_algorithms_dataset_testing.csv')
