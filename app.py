import streamlit as st
import numpy as np
import joblib

# Load the trained model
model_path = 'iris_model.pkl'  
model = joblib.load(model_path)

# Define the Iris species based on the index for displaying the prediction in a human-readable format
iris_species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica', 3: 'Category 3', 4: 'Category 4', 5: 'Category 5'}

# Streamlit application layout
st.title('Iris Flower Species Classifier')

st.write("""
### Please enter the measurements of an Iris flower to predict its species:
""")

# User input fields
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, value=5.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, value=3.5)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, value=1.4)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, value=0.2)

# Predict button
if st.button('Predict'):
    # Reshape the input for the model
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict the Iris species
    prediction = model.predict(input_features)
    prediction_probability = model.predict_proba(input_features)
    
    # Display the prediction
    st.subheader(f'The predicted Iris species is: {iris_species[prediction[0]]}')
    
    # Display the prediction probability
    st.write(f'Prediction Confidence: {np.max(prediction_probability)*100:.2f}%')

st.write("""
## About the Iris Species
The Iris dataset consists of measurements of three different species of the Iris flower:
- **Setosa**: Known for its small size and distinctive petal shape.
- **Versicolor**: Characterized by its medium size and versatile color.
- **Virginica**: Larger in size, with wider petals and vibrant colors.
""")

