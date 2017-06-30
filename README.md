# Deep reinforcement learning in ViZDoom
## Requirements:
- python3
- **Tensorflow** version 1.2 with GPU support
- **ViZDoom** version 1.1.2 (pip install vizdoom)
- [numpy](https://pypi.python.org/pypi/numpy/1.12.0b1)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [ruamel.yaml](https://pypi.python.org/pypi/ruamel.yaml/0.13.4)
- opencv version 3.1.0

To install python dependecies:
```
sudo pip3 install -r requirements.txt
```
## How to use:


### Training (train_a3c.py / train_adqn.py / train_dqn.py):

```bash
./train_a3c.py -s <SETTINGS_FILES>

```
Example:
```bash
./train_a3c.py -s settings/examples/basic_a3c.yml 
```


### Watching (test_a3c.py / test_adqn.py / test_dqn.py):

```bash
./test_a3c.py -h
usage: test_a3c.py [-h] [--settings [YAML_FILE [YAML_FILE ...]]]
                   [--episodes EPISODES_NUM] [--hide-window]
                   [--print-settings] [-fps FRAMERATE] [--agent-view]
                   [--seed SEED]
                   MODEL_FILE

A3C: testing script for ViZDoom

positional arguments:
  MODEL_FILE            Path to trained model.

optional arguments:
  -h, --help            show this help message and exit
  --settings [YAML_FILE [YAML_FILE ...]], -s [YAML_FILE [YAML_FILE ...]]
                        load settings from yaml files. If multiple files are
                        specified, overlapping settings will be overwritten
                        according to order of appearance (e.g. settings from
                        file #1 will be overwritten by file #2). (default:
                        ['settings/basic.yml'])
  --episodes EPISODES_NUM, -e EPISODES_NUM
                        Number of episodes to test. (default: 10)
  --hide-window, -ps    Hide window. (default: False)
  --print-settings, -hw
                        Print settings upon loading. (default: False)
  -fps FRAMERATE        If window is visible, tests will be run with given
                        framerate. (default: 35)
  --agent-view, -aw     If True, window will display exactly what agent
                        sees(with frameskip), not the smoothed out version.
                        (default: False)
  --seed SEED           Seed for ViZDoom. (default: None)

```
Example:
```bash
# You need to have a pretrained model
./train_a3c.py -s models/basic/ACLstmNet_16TH/30June17_20.12 settings/examples/basic_a3c.yml 
```
## Implemented algorithms:
- DQN [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)  
- Double Dueling DQN [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
- A3C [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v2.pdf)
- N-step asynchronous DQN [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v2.pdf)

