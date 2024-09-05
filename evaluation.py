import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model_training import create_dataset
from sklearn.model_selection import train_test_split

def load_model_and_label_encoder(model_name):
    """Load the trained model and label encoder from disk."""
    model = joblib.load(f"{model_name}_model.pkl")
    label_encoder = joblib.load(f"label_encoder-{model_name}.pkl")
    return model, label_encoder

def evaluate_model(model_name, hex_samples):
    """Evaluate a model using various performance metrics."""
    # Load the model and label encoder
    model, label_encoder = load_model_and_label_encoder(model_name)

    # Generate dataset
    dataset = create_dataset(hex_samples)
    X = dataset.iloc[:, :-1].astype(float)  # Features
    y = dataset['Algorithm']  # Labels (Algorithms)
    y_encoded = label_encoder.transform(y)  # Encode labels for evaluation

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Predict using the model
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': accuracy * 100,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'F1 Score': f1 * 100,
        'Confusion Matrix': conf_matrix
    }

def evaluate_all_models(models, hex_samples):
    """Evaluate all specified models and return results in a DataFrame."""
    results = []
    for model_name in models:
        print(f"Evaluating {model_name} model...")
        metrics = evaluate_model(model_name, hex_samples)
        results.append(metrics)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

if __name__ == "__main__":
    models = ["RandomForest", "SVM", "KNN"]
    hex_samples = "cryptographic_algorithms_dataset_testing.csv"

    # Evaluate models and get results
    results_df = evaluate_all_models(models, hex_samples)
    
    # Print the results
    print(results_df)
    
    # Optionally save the results to a CSV file
    results_df.to_csv('model_evaluation_results.csv', index=False)
