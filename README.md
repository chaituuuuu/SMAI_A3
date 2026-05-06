# 🍬 Indian Sweet Recognizer

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-AI-8E75B2.svg)

**Statistical Methods in AI (SMAI) - Assignment 3**
**Theme:** T2 - Indian Food Recognizer
**Variant:** T2.3 (Indian Sweets - 15 Classes)

## 📖 Overview
The **Indian Sweet Recognizer** is an interactive web application that identifies 15 popular Indian sweets (Mithai) from images. Upload a photo, and the app will instantly tell you what dish it is! 

Beyond just classification, the app promotes health awareness by providing:
- 📊 **Nutritional Information**: Approximate calorie counts and macro breakdown.
- ⚠️ **Allergen Badges**: Flags for common allergens like Dairy, Nuts, and Gluten.
- 🏸 **Activity Equivalent**: How much exercise is needed to burn off a serving.
- 🧑‍🍳 **AI Chef (Gemini)**: Integration with Google's Gemini LLM to generate authentic, low-sugar, diabetic-friendly recipe alternatives for the detected sweet.

## 🌟 Features
* **Lightweight Classification Model**: Powered by a PyTorch `MobileNetV3 Small` architecture, fine-tuned specifically on 15 mithai classes for fast and accurate inference.
* **Streamlit Web Interface**: A clean, responsive, and intuitive UI.
* **LLM Integration**: Dynamically prompts the Gemini API to act as a traditional Indian chef to output healthy sugar substitutes and recipes on the fly.

## 🛠️ Local Setup & Installation

**1. Clone the repository and navigate to the directory:**
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd SMAI_A3
```

**2. Install dependencies:**
Make sure you have Python installed. Run the following command to install required packages:
```bash
pip install -r requirements.txt
```

**3. Setup your Gemini API Key:**
To unlock the AI Recipe Generator, you need a Google Gemini API key.
Create a folder named `.streamlit` in the root directory and add a `secrets.toml` file:
```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_actual_api_key_here"
```


**4. Run the Application:**
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

## 📂 Project Structure
- `app.py`: The main Streamlit web application script handling inference, UI, and LLM calls.
- `ModelTraining.ipynb`: The Jupyter Notebook containing the data pipeline, data augmentation, and transfer-learning code used to train the model.
- `mithai_model_best.pth`: The saved PyTorch model weights.
- `metadata.json`: The database mapping the 15 classes to their respective calories, macros, and allergens.
- `requirements.txt`: Python package dependencies.

## 🙏 Acknowledgements
* **Dataset**: Derived from the [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset) on Kaggle.
* **LLM Usage**: The Google Gemini API was utilized within the application to generate custom, context-aware recipe alternatives, as encouraged by the assignment guidelines.
