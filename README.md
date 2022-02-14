# TennisProject
# Udaicity Nanodegree - Deep Reinforcement Learning

## Introduction

The Tennis Project solves the environment with 2 agents. The two agents should collaborate and get a long strike together.

### Rewards
`+0.1` for an agent makes the ball bounce over the net
`-0.01` if the ball falls on the ground or escapes the boundaries

### Actions 
Both agents can performe 2 actions with each action takin a value betwenn between `0` and `1`. This actions reflect the left/right and up/down movement of the agent.

### State
defined by an 8 dimensional vector. Both agents have own observations of the state.

### Goal
Average score of +0.5 over 100 consecutive episodes. For one episode the maximum of both agents scores is taken as the overall episode score.

## Requirements
A detailed description on how to set up the environment and install the needed dependencies can be found on the GitHub page of the [deep reinfocement learning nanodegree](https://github.com/udacity/deep-reinforcement-learning#dependencies).

## Setup Guide
The Unity environment is already prepacked and can be downloaded using the following link:

 - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
 - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
 - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
 - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
