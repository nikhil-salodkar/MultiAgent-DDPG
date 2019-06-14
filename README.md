# Solving MultiAgent Tennis Environment using self play DDPG

## Introduction

In this project we will train two agents in Unity's Tennis environment, each having control of a racket to learn to bounce a ball over a net. The goal of the agents is to keep the ball in play for as long as possible.
The environment and a trained agent looks like this:

![Alt Text](images/tennis.gif)


The simulated environment is Unity based Tennis environment wherein each agent has access to a racket and the aim is to bounce the ball over the net and keep it in play to get maximum reward in each episode.
**A reward of +0.1 is received each time an agent hits the ball over the net. If an agent lets the ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01**. Thus, the goal of agents is to keep the ball in play.
Since both the agents have the same goal they need to work together, hence the problem is a Collaborative MultiAgent environment problem.

The simulated environment provides a simplified **state space having 8 variables** corresponding to position and velocity of the ball and racket. Each agent receives its own local observation.**There are two actions in continuous domain corresponding to movement toward (or away from) the net, and jumping.**

The task is episodic, We will assume the environment solved when the agent is able to achieve at least 0.5 score on average (over 100 consecutive episodes, after taking the maximum over both the agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 potentially different scores. We then take the maximum of these scores.
- This yields a single score for each episode.

## Installation Instructions

1. Download the appropriate Unity environment according to the operating system and decompress the contents into your working directory:
    - Linux: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX : [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit) : [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit) : [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    You can use the above downloaded environment to view how a trained agent behaves in the given environment setup.

2. Since the project is developed using Python and Pytorch some necessary packages need to be installed. Install the necessary libraries and packages mentioned in **requirements.txt**. To install the necessary packages either use pip or create a new conda environment and install the minimum required packages as mentioned in the requirements file. To set up a python environment to run code using conda, you may follow the instructions below:

    Create and activate a new environment with Python 3.6 and install dependencies
    
    - Linux or Mac:
      ```
      conda create --name env-name python=3.6
      source activate env-name
      ```
    
    - Windows:
      ```
      conda create --name env-name python=3.6
      activate env-name
      ```
  
    - Then install the dependecies using 
      ```
      pip install -r requirements.txt
      ```
3. To get more information about unity environments, follow the instructions in this [link](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

4. Run the cells in Jupyter notebook to train a new agent on this environment. Pretrained agent's weights are also present in trained_weights folder.

