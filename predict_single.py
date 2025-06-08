import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
import glob
import os
import sys

# --- Adjustable Parameters ---
# models_dir: Directory containing model files
models_dir = 'models'

# --- Polynomial Model Definition ---
def poly_surface(X, *coeffs):
    x, y, z = X
    if len(coeffs) == 10:  # Degree 2
        a, b, c, d, e, f, g, h, i, j = coeffs
        return a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j
    elif len(coeffs) == 20:  # Degree 3
        a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t = coeffs
        return (a*x**3 + b*y**3 + c*z**3 + d*x**2*y + e*x**2*z + f*y**2*x + g*y**2*z + 
                h*z**2*x + i*z**2*y + j*x*y*z + k*x**2 + l*y**2 + m*z**2 + 
                n*x*y + o*x*z + p*y*z + q*x + r*y + s*z + t)
    else:
        raise ValueError(f"Invalid number of coefficients: {len(coeffs)}. Expected 10 or 20.")

# --- Load Models and Metadata ---
def load_models_and_metadata():
    model_files = glob.glob(os.path.join(models_dir, 'model_*.npy'))
    metadata_files = glob.glob(os.path.join(models_dir, 'region_metadata_*.npy'))
    
    if not model_files or not metadata_files:
        raise FileNotFoundError("Model or metadata files not found in 'models' directory.")
    
    # Load latest metadata
    metadata_file = max(metadata_files, key=os.path.getctime)
    metadata = np.load(metadata_file, allow_pickle=True).item()
    
    models = {}
    for model_file in model_files:
        model_data = np.load(model_file, allow_pickle=True).item()
        region = model_file.split('model_')[1].split('_')[0]
        models[region] = model_data
    
    return models, metadata

# --- Predict with Appropriate Model ---
def predict_with_model(x, y, z, models, metadata):
    # Since we don't have expected_output for new inputs, use only x, y, z for region classification
    # Scale input using metadata scaler (trained on x, y, z, output)
    scaler = StandardScaler()
    scaler.mean_ = np.array(metadata['scaler_mean'])[:3]  # Use only x, y, z means
    scaler.scale_ = np.array(metadata['scaler_scale'])[:3]  # Use only x, y, z scales
    features = np.array([[x, y, z]])
    features_scaled = scaler.transform(features)[0]
    
    # Compute distances to centroids (using only x, y, z dimensions)
    distances = {}
    for region, info in metadata['regions'].items():
        centroid = np.array(info['centroid'])[:3]  # Use only x, y, z dimensions
        distance = np.linalg.norm(features_scaled - centroid)
        distances[region] = distance
    
    # Choose model with closest centroid
    selected_region = min(distances, key=distances.get)
    
    # Scale input for selected model's scaler
    model = models.get(selected_region)
    if model is None:
        print(f"Warning: No model for region {selected_region}. Using general model.", file=sys.stderr)
        selected_region = 'general'
        model = models.get('general')
        if model is None:
            raise ValueError("No general model available.")
    
    model_scaler = StandardScaler()
    model_scaler.mean_ = np.array(model['scaler_mean'])
    model_scaler.scale_ = np.array(model['scaler_scale'])
    input_scaled = model_scaler.transform([[x, y, z]])[0]
    
    # Predict
    prediction = poly_surface(input_scaled, *model['coefficients'])
    
    return prediction

# --- Main Function ---
def main():
    # Check if exactly 3 parameters are provided
    if len(sys.argv) != 4:
        print("Error: predict_single.py requires exactly 3 parameters: trip_duration_days miles_traveled total_receipts_amount", file=sys.stderr)
        sys.exit(1)
    
    # Parse and validate parameters
    try:
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
    except ValueError:
        print("Error: All parameters must be numeric", file=sys.stderr)
        sys.exit(1)
    
    if trip_duration_days < 0 or miles_traveled < 0 or total_receipts_amount < 0:
        print("Error: All parameters must be non-negative", file=sys.stderr)
        sys.exit(1)
    
    # Load models and metadata
    try:
        models, metadata = load_models_and_metadata()
    except Exception as e:
        print(f"Error loading models or metadata: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Predict
    try:
        prediction = predict_with_model(trip_duration_days, miles_traveled, total_receipts_amount, models, metadata)
        print(f"{prediction:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()