# recurrent-ppo-atari
Solving Atari games using PPO with Deep Recurrent networks, for Autonomous and Adaptive Systems class at UNIBO
## About the project

Policy gradients methods are a way to learn a parameterized policy that samples actions without consulting a value function. 
A value function can be approximated and used to help learning the policy. 
This method is often called actor-critic, where actor is a reference to the learned policy and critic refers to the learned value function.

Atari 2600 games are nowadays popular environments where to benchmark various Reinforcement Learning algorithms. 
OpenAI Gym is a framework that allows the user to simply instantiate environments in which to interact. 
It features several types of environments including Atari 2600 games. 
Every Atari game features the same observation and action space, an RGB 210 x 160 frame and 18 possible moves for each time step.

Encoding a state with a single frame is usually not enough to learn a policy able to perform well on Atari games; this is due to the impossibility of extracting from a single frame time-dependent features such as speed and movement. A way to overcome this issue is to stack together several sequential frames and encode them as a single observation. 

One of the aims of this project is to investigate if RNN-based architectures can be effectively used to learn a policy that encodes a single frame as observation, expecting that the RNN unit will "remember" previous frames, helping to extract short time-dependent features.

The algorithm used in this work to solve Atari games is PPO, one of the state-of-the-art policy gradient method. Three different games have been tested with this algorithm: Pong, Breakout and Space Invaders. For each game, both RNN-based and not RNN-based policies have been used and compared one with another. 
In the PPO paper, the authors introduced a simple deep network used to approximate the policy. Another aim of this project is to test that architecture and see if we obtain the same results reported in the paper.

## Requirements
The code has been tested with python 3.10 and requires the libraries listed in requirements.txt:

**Install Modules:** 

```sh
  pip install -U pip
  pip install -r requirements.txt
```
## Usage
To run the algorithm, simply run
```
python main.py <game_name> <cfg_path>
```
Where `<game_name>` must be an Atari game prefix name (list here) while `<cfg_path>` must be the path of a YAML file containing 
the algorithm's hyperparameters, following the structure described in `config/default.yaml`.

After training is finished, in `models/` you can find the actor-critic network's weights. You can evaluate the final model by running
```
python evaluate.py <game_name> <cfg_path> <checkpoint_path>
```
where `<checkpoint_path>` is the path of the file containing the actor-critic network's weights.

If you want additional information, simply run `main.py` or `evaulate.py` adding `--help` as argument.
