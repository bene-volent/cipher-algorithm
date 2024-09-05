import numpy as np
from feature_engineering import generate_features
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Function to test a trained model on new data
def test_model(trained_model, X_test, y_test, label_encoder):
    # Predict the labels for the test set
    y_pred = trained_model.predict(X_test)
    
    # Convert numerical labels back to original labels
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Print evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))

# Function to generate features for new hex input and predict
def predict_new_data(trained_model, new_hex_input):
    # Generate features for the new input
    new_features = generate_features(new_hex_input)
    
    print(new_features)
    data = pd.DataFrame([new_features],columns=['Byte_Freq_' + str(i) for i in range(256)] + ['Entropy', 'Block_Size', 'Byte_Mean', 'Byte_Std'])
    print(data)
    # Predict the cryptographic algorithm
    prediction = trained_model.predict(data)
    
    return prediction[0]
