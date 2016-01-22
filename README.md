# Atari ![Space Invader](http://www.rw-designer.com/cursor-view/74522.png)

**Work In Progress**

Prioritised experience replay [[1]](#references) persistent advantage learning [[2]](#references) dueling [[3]](#references) double [[4]](#references) deep Q-network [[5]](#references) for the Arcade Learning Environment [[6]](#references). Or PERPAL(triple-D)QN for short...

Run `th main.lua` to run headless, or `qlua main.lua` to display the game. The main options are `-game` to choose the ROM (see the [ROM directory](roms/README.md) for more details) and `-mode` as either `train` or `eval`.

In training mode if you want to quit using `Ctrl+C` then this will be caught and you will be asked if you would like to save the DQN agent before quitting. Note that the weights for the network itself are not that large, but the experience replay memory is ~7GB.

## Requirements

Requires [Torch7](http://torch.ch/), and uses CUDA/cuDNN if available. Also requires the following extra luarocks packages:

- torchx
- dpnn
- moses
- logroll
- luaposix
- **classic**
- **xitari**
- **alewrap**
- **rlenvs**

classic, xitari, alewrap and rlenvs can be installed using the following commands:

```sh
luarocks install https://raw.githubusercontent.com/deepmind/classic/master/rocks/classic-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/xitari/master/xitari-0-0.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/alewrap/master/alewrap-0-0.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/rlenvs/master/rocks/rlenvs-scm-1.rockspec
```

## Todo

- Use "sum tree" binary heap for proportional prioritised experience replay

## Acknowledgements

- Georg Ostrovski for confirmation on network usage in advantage operators + note on interaction with Double DQN.

## References

[1] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[2] [Increasing the Action Gap: New Operators for Reinforcement Learning](http://arxiv.org/abs/1512.04860)  
[3] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[4] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[5] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[6] [The Arcade Learning Environment: An Evaluation Platform for General Agents](http://arxiv.org/abs/1207.4708)  
