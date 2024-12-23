import numpy as np
import json
import pickle
from tensorflow.keras.models import load_model
import random
import subprocess  # For calling Flite from the system

# Function to speak the text using Flite
def speak(text):
    """Speak the given text using the Flite command."""
    try:
        # Call Flite via subprocess
        subprocess.run(["flite", "-t", text], check=True)
    except FileNotFoundError:
        print("Error: 'flite' not found. Please ensure it is installed on the system.")
    except Exception as e:
        print(f"Error: {e}")

# Function to load the necessary files (model, vectorizer, label encoder, and responses)
def load_files():
    # Load the trained model
    model = load_model("chatbot_model_LSTM.keras")

    # Load the vectorizer and label encoder
    with open('vectorizer_LSTM.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('label_encoder_LSTM.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Load the responses from the JSON file
    with open('intents.json', 'r') as f:
        data = json.load(f)

    responses = {}
    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']

    return model, vectorizer, label_encoder, responses

# Load the files (model, vectorizer, label encoder, and responses)
model, vectorizer, label_encoder, responses = load_files()

# Function to process user input message
def preprocess_input(message):
    input_vector = vectorizer.transform([message]).toarray()
    input_vector = input_vector.reshape(1, 1, input_vector.shape[1])
    return input_vector

# Store previous interactions
conversation_history = []
provided_responses = set()  # Track provided responses globally across conversations

# Main loop for text-based input
done = False
while not done:
    message = input("You: ")  # Get the message from user input
    if message.lower() == "exit":
        done = True  # Exit condition
        continue

    processed_input = preprocess_input(message)

    # Predict the intent
    predictions = model.predict(processed_input)

    # Get the predicted intent label
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_intent = label_encoder.inverse_transform([predicted_class])[0]

    # Print the assistant's response based on the predicted intent
    print(f"Predicted Intent: {predicted_intent}")

    # Retrieve a random response from the list of responses for the predicted intent
    response_list = responses.get(predicted_intent, ["Sorry, I didn't understand that."])
    response = np.random.choice(response_list)

    print(f"Assistant: {response}")
    speak(response)

    # Store the conversation for context
    conversation_history.append((message, response))

    # Track provided solutions to avoid repetition
    provided_responses.add(response)

    # Ask for another solution in a loop until the user chooses 'no'
    while True:
        follow_up = input("Would you like another solution? (yes/no): ").lower()

        if follow_up == "no":
            print("Assistant: Okay, if you need more help, feel free to ask.")
            speak("Okay, if you need more help, feel free to ask.")
            break  # Exit the loop and stop asking for more solutions

        elif follow_up == "yes":
            # Exclude the already provided responses
            remaining_responses = [resp for resp in response_list if resp not in provided_responses]
            if remaining_responses:
                follow_up_response = f"Another possible solution is: {random.choice(remaining_responses)}."
                # Add the new response to the provided responses set
                provided_responses.add(follow_up_response)
            else:
                follow_up_response = "No other solutions are available at the moment."

            print(f"Assistant: {follow_up_response}")
            speak(follow_up_response)

        else:
            print("Assistant: Please respond with 'yes' or 'no'.")
