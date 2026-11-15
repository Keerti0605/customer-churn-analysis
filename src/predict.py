import pickle
import pandas as pd


def load_model():
    with open("model/churn_model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


def make_prediction(model, input_df):
    try:
        # 🔹 Encode input (same as training)
        input_encoded = pd.get_dummies(input_df)

        # 🔹 Load training columns (IMPORTANT)
        with open("model/features.pkl", "rb") as f:
            features = pickle.load(f)

        # 🔹 Add missing columns
        for col in features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # 🔹 Remove extra columns (if any)
        input_encoded = input_encoded[features]

        # 🔹 Prediction
        prediction = model.predict(input_encoded)
        probability = model.predict_proba(input_encoded)

        return prediction[0], probability[0][1]

    except Exception as e:
        raise Exception(f"Prediction error: {e}")