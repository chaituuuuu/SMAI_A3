import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import google.generativeai as genai

import os

                                    
st.set_page_config(page_title="Indian Sweet Recognizer", page_icon="🍬", layout="centered")

                          
api_configured = False

                                                                                 
try:
    MY_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    MY_GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if MY_GEMINI_KEY and MY_GEMINI_KEY != "PASTE_YOUR_API_KEY_HERE":
    try:
        genai.configure(api_key=MY_GEMINI_KEY)
                                                                                                                                         
        model_name = 'gemini-2.5-flash'
        llm_model = genai.GenerativeModel(model_name) 
        api_configured = True
    except Exception as e:
        st.sidebar.error(f"Error configuring Gemini API: {e}")
else:
    st.sidebar.warning("⚠️ Please paste your Gemini API key at the top of the app.py file to unlock AI recipes.")
                  
@st.cache_data
def load_metadata():
    with open('metadata.json', 'r') as f:
        return json.load(f)

metadata = load_metadata()
class_names = sorted(list(metadata.keys()))

                        
@st.cache_resource
def load_pytorch_model():
    model = models.mobilenet_v3_small(weights=None)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 15)
    model.load_state_dict(torch.load('mithai_model_best.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_pytorch_model()

                                
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

                           
st.title("🍬 Indian Sweet Recognizer (T2.3)")
st.write("Upload a photo of an Indian sweet to analyze its nutritional profile and generate healthy recipe alternatives.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
                       
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.markdown("---")
    
                   
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(img_t)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
        
        predicted_class = class_names[top_catid.item()]
        confidence = top_prob.item() * 100
        
                                    
    formatted_name = predicted_class.replace('_', ' ').title()
    st.subheader(f"Prediction: {formatted_name}")
    st.caption(f"Network Confidence: {confidence:.2f}%")
    
    item_data = metadata[predicted_class]
    calories = item_data['calories']
    
                                            
    st.write("### Nutritional Breakdown (Per 100g Serving)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Calories", value=f"{calories} kcal")
    with col2:
        st.metric(label="Carbs", value=f"{item_data['macros']['carbs']}g")
    with col3:
        st.metric(label="Fat", value=f"{item_data['macros']['fat']}g")
    with col4:
        st.metric(label="Protein", value=f"{item_data['macros']['protein']}g")

    st.write("")           
    
                                      
    st.write("**Allergens Detected:**")
    badges_html = ""
    for allergen in item_data['allergens']:
        if allergen == "Dairy":
            badges_html += f'<span style="background-color: #cce5ff; color: #004085; padding: 6px 12px; border-radius: 15px; margin-right: 5px; font-weight: 500;">🥛 {allergen}</span>'
        elif allergen == "Nuts":
            badges_html += f'<span style="background-color: #fff3cd; color: #856404; padding: 6px 12px; border-radius: 15px; margin-right: 5px; font-weight: 500;">🥜 {allergen}</span>'
        elif allergen == "Gluten":
            badges_html += f'<span style="background-color: #f8d7da; color: #721c24; padding: 6px 12px; border-radius: 15px; margin-right: 5px; font-weight: 500;">🌾 {allergen}</span>'
    
    if badges_html:
        st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.write("None detected.")
        
    st.markdown("---")
    st.write("### 🏸 Activity Equivalent")
                                                                                      
    badminton_mins = int(calories / 8.5)
    st.info(f"**Burn it off:** You would need to play **{badminton_mins} minutes** of competitive badminton to burn off one serving of {formatted_name}.")
                                                  
    st.write("### 🧑‍🍳 Healthy Recipe Generator")
    st.write("Want to enjoy this sweet without the sugar crash? Ask our AI Chef for a diabetic-friendly alternative.")
    
    if api_configured:
        if st.button(f"Generate Low-Sugar {formatted_name} Recipe"):
            with st.spinner(f"Chef Gemini is crafting a healthy {formatted_name} recipe..."):
                try:
                                                                 
                    prompt = f"""
                    You are an expert nutritionist and traditional Indian chef. 
                    The user wants to eat a traditional Indian sweet called '{formatted_name}'.
                    Provide a diabetic-friendly, low-sugar recipe alternative for this sweet. 
                    Suggest healthy sugar substitutes (like stevia, monk fruit, or dates where appropriate).
                    Format the output cleanly using Markdown with these headers:
                    - **Why this is healthier:** (Brief explanation)
                    - **Ingredients:** (Bullet points)
                    - **Instructions:** (Numbered steps)
                    Keep it authentic but healthy!
                    """
                    
                                                             
                    response = llm_model.generate_content(prompt)
                    
                                                                                     
                    with st.expander("✨ Your AI-Generated Recipe ✨", expanded=True):
                        st.markdown(response.text)
                        
                except Exception as e:
                    st.error(f"Oops! Something went wrong with the AI: {e}")