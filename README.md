# Atari ![Space Invader](http://www.rw-designer.com/cursor-view/74522.png)

**Work In Progress**

Dueling [[1]](#references) Double [[2]](#references) DQN [[3]](#references) with prioritised experience replay [[4]](#references) for the Arcade Learning Environment [[5]](#references).

Run `th main.lua` to run headless, or `qlua main.lua` to display the game. The main options are `-game` to choose the ROM (see the [ROM directory](roms/README.md) for more details) and `-mode` as either `train` or `eval`.

## Requirements

Requires [Torch7](http://torch.ch/), and uses CUDA/cuDNN if available. Also requires the following extra packages:

- dpnn
- moses
- logroll
- classic
- xitari
- alewrap

classic, xitari and alewrap can (hopefully) be installed using the following commands:

```sh
luarocks install https://raw.githubusercontent.com/deepmind/classic/master/rocks/classic-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/xitari/master/xitari-0-0.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/alewrap/master/alewrap-0-0.rockspec
```

## Todo

- **Use 4 frames (instead of 1)...**
- Implement rank-based prioritised experience replay
- Use "sum tree" binary heap for proportional prioritised experience replay

## References

[1] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[2] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[3] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[4] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[5] [The Arcade Learning Environment: An Evaluation Platform for General Agents](http://arxiv.org/abs/1207.4708)  
~~[6] [Increasing the Action Gap: New Operators for Reinforcement Learning](http://arxiv.org/abs/1512.04860)~~  
