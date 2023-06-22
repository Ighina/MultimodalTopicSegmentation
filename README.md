# Audio-Topic-Segmentation
Repository for the paper "Exploring pre-trained Audio Neural Representations for Audio Topic Segmentation"

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

## NonNews/-SBBC
Follow the instructions inside the README.md file in the NonNews-BBC folder in this repository.

## RadioNews-SBBC
Follow the instructions inside the README.md file in the RadioNews-BBC folder in this repository.

# Predict
In order to use the pretrained model to segment input audio files, use predict.py with custom arguments.

An example usage is included below, using the provided model pre-trained on NonNews-BBC dataset with OpenL3 embeddings with last pooling.

`python python predict.py -ee -ef openl3 -hyp pretrained_model/results.txt -model pretrained_model/best_model -exp first_trial -gpus 1 -v -af example_inputs`

Adjust the -gpus argument (set it to 0 if you don't have a GPU) and include an mp3 file in the example_inputs directory to use the above example. The output will be available in the first_trial/audio_segments directory, in case the model was able to identify at least one topic boundary. Otherwise, that directory will be empty and a warning message will be printed.
