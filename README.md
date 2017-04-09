# Deep reinforcement learning in ViZDoom
## Requirements:
- python3
- **Tensorflow** version 0.12r with GPU support
- **ViZDoom** version 1.1.0rc1+ (pip install vizdoom)
- [numpy](https://pypi.python.org/pypi/numpy/1.12.0b1)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [ruamel.yaml](https://pypi.python.org/pypi/ruamel.yaml/0.13.4)

To install python dependecies:
```
sudo pip3 install numpy tqdm ruamel.yaml 
```

## Implemented algorithms:
- DQN [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)  
- Double Dueling DQN [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
- A3C [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v2.pdf)
- N-step asynchronous DQN [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v2.pdf)

# TODO:
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952v4.pdf)
