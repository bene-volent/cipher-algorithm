# In your main script, train a model and test it
from model_training import train_model
from model_testing import test_model,predict_new_data

# Step 1: Train the model
trained_model,label_encoder = train_model("RandomForest",hex_samples='cryptographic_algorithms_dataset_testing.csv')  # Choose RandomForest, SVM, or KNN


# Step 2: Test the trained model on a new hex input
new_hex_input = "4D 0F D0 D2 A0 09 F5 10 E0 8A 30 06 4D 53 A4 1F 63 4A 90 29"
predicted_algo = predict_new_data(trained_model, new_hex_input)

print(f"Predicted Algorithm: {predicted_algo} : {label_encoder.inverse_transform([predicted_algo])[0]}")
