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

## Implemented algorithms:
- DQN [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)  
- Double Dueling DQN [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
- A3C [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v2.pdf)
- N-step asynchronous DQN [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v2.pdf)

## How to use:

### Settings
Alll training scripts looad settings from multiple yaml file (settings are combined, reocurring keys are overwritten by newest entries). By default "settings/defaults/common_defaults.yml" and "settings/XXX_defaults.yml will be loaded (XXX in {a3c, adqn, dqn}). To load addditional settings use **-s / --settings** switch.

For convenience, multiple yml files with scenarios are held separately.

>>> Default settings aren't particularly focused on giving fast results and might leave no output for very long time. Settings in settings/examples directory should work out of the box though.

### Training (train_a3c.py / train_adqn.py / train_dqn.py):

```bash
./train_a3c.py -s <SETTINGS_FILES>

```
## Example:
```bash
./train_a3c.py -s settings/examples/basic_a3c.yml 

# Using your settingo:
./train_a3c.py -s {YOUR_SETTINGS1} {YOUR_SETTINGS2} settings/basic.yml 
```
## Output:
>>> Tensorboard logger level is set to 2 by defulat so don't expect info logs from tf.

- Lots of console output including loaded settings, training/test results and errors. The output is partly colored so it might be difficulat to read as raw text.
- Log file with output hte same as one from console in a path resembling {logfile}_{DATE_AND_TIME}.log. (if logfile is specified)
- Tensorboard scalar summaries with scores (min/mean/max/std) and learning rate in {tf_logdir} (tensorboard_logs by default).
- TF model saved in a path resembling {models_path}/{scenario_tag}/{NETWORK_NAME}/{DATE_AND_TIME}

### Watching (test_a3c.py / test_adqn.py / test_dqn.py):

```bash
usage: test_a3c.py [-h] [--episodes EPISODES_NUM] [--hide-window]
                   [--print-settings] [--fps FRAMERATE] [--agent-view]
                   [--seed SEED] [-o STATS_OUTPUT_FILE]
                   [--deterministic DETERMINISTIC]
                   MODEL_PATH

A3C: testing script for ViZDoom

positional arguments:
  MODEL_PATH            Path to trained model directory.

optional arguments:
  -h, --help            show this help message and exit
  --episodes EPISODES_NUM, -e EPISODES_NUM
                        Number of episodes to test. (default: 10)
  --hide-window, -ps    Hide window. (default: False)
  --print-settings, -hw
                        Print settings upon loading. (default: False)
  --fps FRAMERATE, -fps FRAMERATE
                        If window is visible, tests will be run with given
                        framerate. (default: 35)
  --agent-view, -av     If True, window will display exactly what agent
                        sees(with frameskip), not the smoothed out version.
                        (default: False)
  --seed SEED, -seed SEED
                        Seed for ViZDoom. (default: None)
  -o STATS_OUTPUT_FILE, --output STATS_OUTPUT_FILE
                        File for output of stats (default: None)
  --deterministic DETERMINISTIC, -d DETERMINISTIC
                        If 1 dtests will be deterministic. (default: 1)


```
## Example:
```bash
# You need to have a pretrained model
./test_a3c.py models/basic/ACLSTMNet/09.04_11-21 -e 10 --seed 123
```

