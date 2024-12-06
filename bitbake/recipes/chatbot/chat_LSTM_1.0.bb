# chatbot_1.0.bb

SUMMARY = "A simple chatbot that uses an LSTM model for predictions."
DESCRIPTION = "A simple chatbot that uses an LSTM model to predict user intent based on trained data."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=xxxxx"  # Adjust with the correct checksum if you include a LICENSE file.

# Specify the GitHub repository URL where the chatbot project is hosted
SRC_URI = "git://github.com/Ayatelzook/chatbot-project.git;branch=master;protocol=https"

# Add any required dependencies (if needed)
DEPENDS = "python3-pip"

# Define the directory to install files to
S = "${WORKDIR}/git"

# Installation steps
do_install() {
    # Create necessary directories for installation
    install -d ${D}${bindir}  # Directory for executable files
    install -d ${D}${sysconfdir}/chatbot  # Directory for chatbot configuration
    install -d ${D}${sysconfdir}/chatbot/model  # Directory for the model
    install -d ${D}${sysconfdir}/chatbot/vectorizer  # Directory for vectorizer
    install -d ${D}${sysconfdir}/chatbot/label_encoder  # Directory for label encoder
    install -d ${D}${sysconfdir}/chatbot/intents  # Directory for intents

    # Install the chatbot Python script
    install -m 755 ${S}/scripts/chatbot.py ${D}${bindir}/chatbot.py

    # Install model files (LSTM model, vectorizer, label encoder)
    install -m 644 ${S}/model/chatbot_model_LSTM.keras ${D}${sysconfdir}/chatbot/model/chatbot_model_LSTM.keras
    install -m 644 ${S}/vectorizer/vectorizer_LSTM.pkl ${D}${sysconfdir}/chatbot/vectorizer/vectorizer_LSTM.pkl
    install -m 644 ${S}/label_encoder/label_encoder_LSTM.pkl ${D}${sysconfdir}/chatbot/label_encoder/label_encoder_LSTM.pkl

    # Install the training data (intents.json)
    install -m 644 ${S}/intents/intents.json ${D}${sysconfdir}/chatbot/intents.json
}

# Optionally, you can define the Python environment to run the chatbot (if needed)
do_install_append() {
    # Install Python dependencies if necessary
    pip3 install tensorflow pyttsx3
}
