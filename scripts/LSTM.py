import pyttsx3  # For text-to-speech conversion
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import LabelEncoder
import random
import pickle

# Initialize the speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)


# Function to speak the text
def speak(text):
    """Speak the given text using the pyttsx3 engine."""
    engine.say(text)
    engine.runAndWait()


# Function to preprocess and reshape the JSON data
def preprocess_data(json_file):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'intents' in data:
        intents = data['intents']
        patterns = []
        tags = []
        responses = {}
        for intent in intents:
            for pattern in intent['patterns']:
                patterns.append(pattern)
                tags.append(intent['tag'])
            responses[intent['tag']] = intent['responses']

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(patterns).toarray()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(tags)

    else:
        raise ValueError("Expected data to be a dictionary with 'intents' key.")

    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X, np.array(y), label_encoder, vectorizer, responses


# Preprocess data (using 'intents.json')
X, y, label_encoder, vectorizer, responses = preprocess_data('intents.json')

# Build the model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, batch_size=5, verbose=1)

# Save the trained model
model.save("chatbot_model_LSTM.keras")

# Save vectorizer and label encoder for later use
with open('vectorizer_LSTM.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('label_encoder_LSTM.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


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
