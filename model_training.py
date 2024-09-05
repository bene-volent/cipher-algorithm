import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from feature_engineering import generate_features
from sklearn.preprocessing import LabelEncoder
import joblib


label_encoder = LabelEncoder()



# Function to generate dataset from CSV file
def create_dataset(filename: str) -> pd.DataFrame:
    # Load CSV file into DataFrame
    df = pd.read_csv(filename)
    
    data = []
    for _, row in df.iterrows():
        hex_input = row['Cipher']
        algo = row['Algorithm']
        print((_ * 100) //df.shape[0])
        features = generate_features(hex_input)
        data.append(np.append(features, algo))  # Append features and the label
    
    # Create DataFrame
    features = [d[:-1] for d in data]  # Features
    labels = [d[-1] for d in data]     # Labels
    dataset = pd.DataFrame(features,columns=['Byte_Freq_' + str(i) for i in range(256)] + ['Entropy', 'Block_Size', 'Byte_Mean', 'Byte_Std'])
    dataset['Algorithm'] = labels
    return dataset

# Function to train model
def train_model(algorithm="RandomForest", hex_samples="cryptographic_algorithms_dataset.csv"):
    # Generate dataset
    dataset = create_dataset(hex_samples)
    X = dataset.iloc[:, :-1].astype(float)  # Features
    y = dataset['Algorithm']  # Labels (Algorithms)
    y_encoded = label_encoder.fit_transform(y)
    
    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Choose and train the model
    if algorithm == "RandomForest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == "SVM":
        clf = SVC(kernel='linear')
    elif algorithm == "KNN":
        clf = KNeighborsClassifier(n_neighbors=3)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    clf.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(clf, f"{algorithm}_model.pkl")
    joblib.dump(label_encoder, f"label_encoder-{algorithm}.pkl")
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{algorithm} Model Accuracy: {accuracy * 100:.2f}%")
    
    return clf,label_encoder  # Return the trained model for later use


if __name__=='__main__':
    print(create_dataset('cryptographic_algorithms_dataset_testing.csv').head())