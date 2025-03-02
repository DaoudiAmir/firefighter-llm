# Firefighter Assistant AI - Interactive App

This Streamlit application provides an interactive interface to the fine-tuned Firefighter Assistant Language Model.

## Features

- **Interactive Chat Interface**: Ask questions and get responses from the AI assistant
- **Predefined Prompts**: Test the model with carefully crafted prompts related to firefighting
- **Model Selection**: Choose between the final model or the best model (lowest validation loss)
- **Adjustable Parameters**: Control temperature and response length
- **Device Selection**: Run on CPU or GPU (if available)
- **Conversation History**: Review previous exchanges

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements-app.txt
```

2. Make sure you have the trained model in the correct location:
   - Final model: `models/firefighter-assistant-v1/final_model`
   - Best model: `models/firefighter-assistant-v1/best_model`

## Running the App

```bash
streamlit run app.py
```

This will start the Streamlit server and open the app in your default web browser.

## Usage Tips

1. **Predefined Prompts**: Use the sidebar to select from predefined prompts to test specific scenarios
2. **Custom Queries**: Enter your own instructions and context in the input fields
3. **Temperature**: Adjust the temperature slider to control randomness (lower for more deterministic responses)
4. **Response Length**: Control the maximum length of generated responses

## Model Information

This model is based on DeepSeek-R1-Distill-Qwen-1.5B and has been fine-tuned specifically for firefighting and emergency response scenarios. It can provide guidance on:

- Fire safety protocols
- Emergency response procedures
- First aid instructions
- Evacuation planning
- Equipment usage
- Hazard management
- Communication during emergencies

## Limitations

- The model was trained on a limited dataset and may not cover all firefighting scenarios
- Always verify critical safety information with official sources
- The model is not a replacement for professional training or emergency services
