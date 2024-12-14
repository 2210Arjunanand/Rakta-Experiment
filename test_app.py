import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load models
exception_model = load_model('Exception_model_20Nov.h5')
xception_model = load_model('xception_blood_cancer_model.h5')

# Folder containing images
image_folder = r"D:\Raktcure\Png testing\Png"  # Replace with the correct path
output_file = "model_predictions.xlsx"

# Initialize a list to store results
results = []

# Process each image in the folder
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # First model prediction
        exception_prediction = exception_model.predict(img_array)[0][0]
        if exception_prediction > 0.5:  # Pass to second model if condition met
            final_prediction = xception_model.predict(img_array)[0][0]
            predicted_class = "Positive" if final_prediction > 0.5 else "Negative"
            confidence = final_prediction
        else:
            predicted_class = "Not Accepted by Exception Model"
            confidence = exception_prediction

        # Append result
        results.append({
            "Image": img_name,
            "Prediction": predicted_class,
            "Confidence": round(confidence, 4),
        })

    except Exception as e:
        # Append error details if an image fails processing
        results.append({
            "Image": img_name,
            "Prediction": "Error",
            "Confidence": str(e),
        })

# Save results to Excel
df = pd.DataFrame(results)
df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")