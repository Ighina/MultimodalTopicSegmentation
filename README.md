# Multimodal Topic Segmentation
Repository for the paper [Multimodal Topic Segmentation of Podcast Shows with Pre-trained Neural Encoders](https://dl.acm.org/doi/10.1145/3591106.3592270)

# Installation
Before using this repository, create a virtual environment such as:

`virtualenv audio_topic_seg`

Then, activate it:

`source audio_topic_seg/bin/activate`

And, from inside the environment, install all the dependencies with:

`pip install -r requirements.txt`

Note: for faster embeddings extraction with OpenL3, it is suggested to install tensorflow with gpu capabilities by further running:
`pip install tensorflow-gpu`

# Use
To replicate the results for the individual datasets presented in the original paper, follow below instructions

## NonNews-SBBC
Follow the instructions inside the README.md file in the NonNews-BBC folder in this repository.

## RadioNews-SBBC
Follow the instructions inside the README.md file in the RadioNews-BBC folder in this repository.

# Predict
In order to use the pretrained model to segment input audio files, use predict.py with custom arguments.
