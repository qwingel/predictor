import pickle
import numpy as np
import pandas as pd

"""
Prediction script for new match data
Uses the optimized Logistic Regression model
"""

def load_model(model_path='model.pkl'):
    """Load the trained model and feature indices"""
    with open(model_path, 'rb') as f:
        model, features_optimized = pickle.load(f)
    return model, features_optimized

def predict_match(delta_rating, map_id, delta_map_wr_windowed,
                  map_missing_flag, LAN_flag, h2h_shrunk, h2h_count):
    """
    Predict match outcome

    Parameters:
    -----------
    delta_rating : float
        Rating difference (team2 - team1)
    map_id : int
        Map identifier (not used in optimized model)
    delta_map_wr_windowed : float
        Map winrate difference (not used in optimized model)
    map_missing_flag : int (0 or 1)
        Whether map data is missing
    LAN_flag : int (0 or 1)
        Whether match is on LAN
    h2h_shrunk : float
        Head-to-head history (shrunk)
    h2h_count : int
        Number of head-to-head matches

    Returns:
    --------
    prediction : int (-1 or 1)
        -1: Team 1 wins, 1: Team 2 wins
    probability : float
        Probability of predicted outcome
    """
    # Load model
    model, features_optimized = load_model()

    # Create feature vector (all 7 features)
    X = np.array([[delta_rating, map_id, delta_map_wr_windowed,
                   map_missing_flag, LAN_flag, h2h_shrunk, h2h_count]])

    # Select optimized features
    X_optimized = X[:, features_optimized]

    # Predict
    prediction = model.predict(X_optimized)[0]
    probabilities = model.predict_proba(X_optimized)[0]
    confidence = probabilities[1] if prediction == 1 else probabilities[0]

    return int(prediction), float(confidence)

def predict_from_csv(csv_path):
    """
    Predict outcomes for matches in a CSV file

    Parameters:
    -----------
    csv_path : str
        Path to CSV file with columns: delta_rating, map_id, delta_map_wr_windowed,
        map_missing_flag, LAN_flag, h2h_shrunk, h2h_count

    Returns:
    --------
    DataFrame with predictions and probabilities
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Load model
    model, features_optimized = load_model()

    # Select optimized features
    X = df.values
    X_optimized = X[:, features_optimized]

    # Predict
    predictions = model.predict(X_optimized)
    probabilities = model.predict_proba(X_optimized)

    # Add to dataframe
    df['prediction'] = predictions
    df['confidence'] = np.max(probabilities, axis=1)
    df['prob_team1_wins'] = probabilities[:, 0]
    df['prob_team2_wins'] = probabilities[:, 1]

    return df

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("MATCH PREDICTION EXAMPLE")
    print("="*70)

    # Example match
    delta_rating = -22
    map_id = 4
    delta_map_wr_windowed = 0.0
    map_missing_flag = 0
    LAN_flag = 1
    h2h_shrunk = 0
    h2h_count = 0

    prediction, confidence = predict_match(
        delta_rating, map_id, delta_map_wr_windowed,
        map_missing_flag, LAN_flag, h2h_shrunk, h2h_count
    )

    winner = "Team 1" if prediction == -1 else "Team 2"
    print(f"\nMatch Details:")
    print(f"  Delta Rating: {delta_rating}")
    print(f"  Map Missing: {map_missing_flag}")
    print(f"  LAN Match: {LAN_flag}")
    print(f"  H2H Shrunk: {h2h_shrunk}")
    print(f"  H2H Count: {h2h_count}")
    print(f"\nPrediction: {winner} wins")
    print(f"Confidence: {confidence:.2%}")
    print("="*70)
