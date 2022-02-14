# Report
## The learning approach

To solve the descriped environment a deep deterministic policy gradient (DDPG) algorithm is deployed and extended to the multi-agent scenario. This approach is very similar to DQNs and can be used in contious action spaces like the four continous action variables we have in this environment. The actor of the DDPG learns a (deterministic) policy directly predicting the best possible action at state s. The cretic is trained to approximate the optimal action value function and evaluate it for the best action chosen by the actor.

Similar to the DQN the DDPG also uses a replay buffer and a target network for a more stable trainig process. This reply buffer is sharedbetween the two agents.

The DDPG algorithm learns a deterministic policy to Gaussian noise is added to also allow exploratory characteristics.

## Implementation
The implementation is based on the one in the privious project (https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal). The code is addapted to the unity environment and changed for multiple agents. A `multi_agent` class is created that functions as a share wrapper for the two agents. This class also contains the shared replay buffer.

The hyperparameters are tuned to evaluate their impact on training performance.

## Hyperparameter
The model contains the following hyperparamter. the parameter value are based on the previous project in the nanodegree and yield good results.

### Layers of the DNN
The actor and the critic use a three layer architecture. The actor has 512 units in the first two layers and 256 units in the last one. Relu functions are used for activation in the first and second layer and the output neuron uses tanh function.

The critic also has three layers with the chosen action concatenated with the output of the first layer.

### Other Parameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512       # minibatch size
GAMMA = 0.99            # 0.99 discount factor
LEARN_EVERY = 1        # Update the networks 1 times after every 1 timesteps
LEARN_NUMBER = 1       # Update the networks 1 times after every 1 timesteps

GAMMA = 0.99            # 0.99 discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # 2e-4 learning rate of the actor # ADDED
LR_CRITIC = 1e-4        # 2e-4 learning rate of the critic # ADDED
WEIGHT_DECAY = 0        # L2 weight decay

## Learning Curve
![Reward curve](learning.png?raw=true)

## Further improvements
A next step would be a more structured evaluation of the hyperparameter space. Currently the optimal parameters were optained through "educated" guessing. Especially the evaluation of the impact of the `LEARN_EVERY` and `LEARN_NUMBER` parameter schould be extended. This could smooth the learning of the algorithm.