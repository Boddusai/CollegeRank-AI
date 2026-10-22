import joblib
import pandas as pd


MODEL_FILE = "models/college_rank_model.pkl"


# Load model
model = joblib.load(MODEL_FILE)

print("=" * 70)
print("COLLEGERANK-AI PREDICTION TEST")
print("=" * 70)


# Example input
input_data = pd.DataFrame([
    {
        "year": 2026,
        "round": 2,
        "institute_type": "IIT",
        "institute": "Indian Institute  of Technology Bombay",
        "academic_program": "Computer Science and Engineering (4 Years, Bachelor of Technology)",
        "quota": "AI",
        "seat_type": "OPEN",
        "gender": "Gender-Neutral",
        "opening_rank": 1,
    }
])


# Prediction
prediction = model.predict(input_data)

predicted_closing_rank = round(float(prediction[0]))


print("\nInput:")
print(input_data.to_string(index=False))

print("\nPredicted closing rank:")
print(predicted_closing_rank)

print("\nPrediction completed successfully.")