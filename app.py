import streamlit as st
import pandas as pd
import joblib  # Wapas joblib use kar rahe hain kyunki aapka model joblib se bana hai

# 1. Saved Model ko joblib ke zariye load karein
model = joblib.load('starbucks_decision_tree.pkl')

# 2. App ka Title aur Header
st.set_page_config(page_title="Starbucks Drink Predictor", page_icon="☕", layout="centered")
st.title("☕ Starbucks Drink Category Predictor")
st.write("Apni drink ke nutrients enter karein aur Machine Learning model aapko category batayega!")

st.markdown("---")

# 3. User Input Fields (Sliders)
st.subheader("📊 Drink Nutrients Data")

calories = st.slider("Calories (kcal)", min_value=0, max_value=600, value=150, step=5)
sugar = st.slider("Sugars (grams)", min_value=0, max_value=100, value=20, step=1)
fat = st.slider("Total Fat (grams)", min_value=0, max_value=50, value=5, step=1)

st.markdown("---")

# 4. Prediction Button
if st.button("🔮 Predict Category", type="primary"):
    # Input ko dataframe me badlein
    input_data = pd.DataFrame([[calories, sugar, fat]], columns=['Calories', 'Sugars (g)', 'Total Fat (g)'])
    
    # Model prediction
    prediction = model.predict(input_data)
    
    # Result display
    st.balloons()
    st.success(f"🎯 Model Prediction: **{prediction[0]}**")
