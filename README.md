# Atari ![Space Invader](http://www.rw-designer.com/cursor-view/74522.png)

**Work In Progress**

Prioritised experience replay [[1]](#references) persistent advantage learning [[2]](#references) bootstrapped [[3]](#references) dueling [[4]](#references) double [[5]](#references) deep Q-network [[6]](#references) for the Arcade Learning Environment [[7]](#references). Or PERPALB(triple-D)QN for short...

Run `th main.lua` to run headless, or `qlua main.lua` to display the game. The main options are `-game` to choose the ROM (see the [ROM directory](roms/README.md) for more details) and `-mode` as either `train` or `eval`. Can visualise saliency maps [[8]](#references), optionally using guided [[9]](#references) or "deconvnet" [[10]](#references) backpropagation. Saliency map modes are applied at runtime so that they can be applied retrospectively to saved models.

To run experiments based on hyperparameters specified in the individual papers, use `./run.sh <paper> <game> <args>`. For more details see the script itself. By default the code trains on a demo environment called Catch - use `./run.sh demo` to run the demo with good default parameters.

In training mode if you want to quit using `Ctrl+C` then this will be caught and you will be asked if you would like to save the agent. Note that this includes a copy the experience replay memory, so will total ~7GB. The main script also automatically saves the weights of the best performing DQN (according to the average validation score).

In evaluation mode you can create recordings with `-record true` (requires FFmpeg); this does not require using `qlua`. Recordings will be stored in the videos directory.

## Requirements

Requires [Torch7](http://torch.ch/), and uses CUDA if available. Also requires the following extra luarocks packages:

- luaposix
- moses
- logroll
- classic
- torchx
- dpnn
- nninit
- **xitari**
- **alewrap**
- **rlenvs**

xitari, alewrap and rlenvs can be installed using the following commands:

```sh
luarocks install https://raw.githubusercontent.com/Kaixhin/xitari/master/xitari-0-0.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/alewrap/master/alewrap-0-0.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/rlenvs/master/rocks/rlenvs-scm-1.rockspec
```

## Acknowledgements

- Georg Ostrovski for confirmation on network usage in advantage operators + note on interaction with Double DQN.

## References

[1] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[2] [Increasing the Action Gap: New Operators for Reinforcement Learning](http://arxiv.org/abs/1512.04860)  
[3] [Deep Exploration via Bootstrapped DQN](http://arxiv.org/abs/1602.04621)  
[4] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[5] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[6] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[7] [The Arcade Learning Environment: An Evaluation Platform for General Agents](http://arxiv.org/abs/1207.4708)  
[8] [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](http://arxiv.org/abs/1312.6034)  
[9] [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806)  
[10] [Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901)  
