ECE697DL: Deep Learning on IOT Devices

Spring 2022

Jonah O'Brien Weiss, Abhinaba Das, Subhankar Chowdhury

jobrienweiss@umass.edu, abhinabadas@umass.edu, subhankarcho@umass.edu

# Reinforcement Learning Connect4 Agent

This repo contains code to train a reinforcement learning agent to play the board game
Connect4.  The agent can be trained in pytorch or tensorflow.  It can then be converted
to tensorflowlite, where it can be deployed on the google coral edge device.  The agent
can interface with a GUI so a user can play against it.

## Directory Structure

Notable files are outlined below.

```yaml
GUI/:
  tensorflow_model/     # same files as from folder tensorflow_model below.
  gui_against_ai.py     # file to play against a reinforcement learning agent.
  # supporting files
pytorch_model/:
  all_code.py           # python code to train/implement a pytorch RL model.
                        # NOTE - this code is only for reading, not to run,
                        # requires rewrite to run.
  connect_X.ipynb       # Jupyter Notebook version of above with explanations.
                        # this code can be run in google colab with GPU.
tensorflow_model/:
  run/                  # where new models and logs are stored.
  run_archive/          # where previous models and logs are stored.
  agent.py              # User player or RL agent
  config.py             # training hyperparameters
  convert_model.py      # convert model from tensorflow to tflite
  funcs.py              # play games between 2 agents
  game.py               # array representation of the game state
  main.py               # run this to start training
  MCTS.py               # implements Monte Carlo Tree Search for Connect4
  model_tournament.py   # plays games between tensorflow model and pytorch model.
  model.py              # sets up the CNN.
  pytorch_model.py      # sets up the pytorch model from folder above.
                        # NOTE - model must be downloaded from 
                        # (https://github.com/neoyung/connect-4)
  # supporting files
```

## Installation and Requirements

ML training was run on Ubuntu linux with Python 3.9 and an NVIDIA Quadro RTX 8000 GPU.
Model can be deployed on any OS without a GPU and on the Google Coral.

Install requirements with 
```
pip3 install -r requirements.txt
```

## Running the Code to Play Against the Agent

There are several options to run the GUI depending on the type of 
system that you have.

Configuration of the game is in ./GUI/tensorflow_model/config.py.
Increasing the MCTS_SIMS will result in a model that is harder to beat,
but will take longer to make decisions.

### Run without GPU

Use this option to run using tensorflow lite on a CPU.  This will allow the user to
play against a trained version of the tensorflow agent.  The model version is taken 
from the function setup_ai() in ./GUI/tensorflow_model/ai_player.py.

Note that the variable TFLITE in ./GUI/tensorflow_model/config.py must be True
and TPU must be false.

Run
```
python3 ./GUI/gui_against_ai.py
```

### Run with GPU

If your system has a GPU, you can set TFLITE in ./GUI/tensorflow_model/config.py to be false to 
run with tensorflow, or you can set it to true to run with tflite (which is faster). 
TPU must be false.

Run
```
python3 ./GUI/gui_against_ai.py
```

### Run on Google Coral (TPU)

Follow the steps to set up the Google Coral (https://www.coral.ai/docs/dev-board/get-started/#requirements).
This takes a few hours.

Then ensure that the Google Coral has internet connection (through ethernet or wifi), and install the 
requirements as stated above.

Set TFLITE in ./GUI/tensorflow_model/config.py to be True and set TPU to True.

Connect the Google Coral to a computer monitor via HDMI (note-not all hdmi connections will work, for example
we tried to use an hdmi connection to a projector and it did not work).

Run
```
python3 ./GUI/gui_against_ai.py
```
