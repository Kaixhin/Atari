# Atari ![Space Invader](http://www.rw-designer.com/cursor-view/74522.png)

**Work In Progress:** Crossed out items have been partially implemented.

~~Prioritised experience replay~~ [[1]](#references) persistent advantage learning [[2]](#references) ~~bootstrapped~~ [[3]](#references) dueling [[4]](#references) double [[5]](#references) deep ~~recurrent~~ [[6]](#references) Q-network [[7]](#references) for the Arcade Learning Environment [[8]](#references) with asynchronous[[12]](#references) modes. Or PERPALB(triple-D)RQN for short...

Run `th main.lua` to run headless, or `qlua main.lua` to display the game. The main options are `-game` to choose the ROM (see the [ROM directory](roms/README.md) for more details) and `-mode` as either `train` or `eval`. Can visualise saliency maps [[9]](#references), optionally using guided [[10]](#references) or "deconvnet" [[11]](#references) backpropagation. Saliency map modes are applied at runtime so that they can be applied retrospectively to saved models.

To run experiments based on hyperparameters specified in the individual papers, use `./run.sh <paper> <game> <args>`. `<args>` can be used to overwrite arguments specified earlier (in the script); for more details see the script itself. By default the code trains on a demo environment called Catch - use `./run.sh demo` to run the demo with good default parameters. Note that `main.lua` uses CUDA by default if available, but the Catch network is small enough that it runs faster on CPU.

In training mode if you want to quit using `Ctrl+C` then this will be caught and you will be asked if you would like to save the agent. Note that this includes a copy the experience replay memory, so will total ~7GB. The main script also automatically saves the weights of the best performing DQN (according to the average validation score).

In evaluation mode you can create recordings with `-record true` (requires FFmpeg); this does not require using `qlua`. Recordings will be stored in the videos directory.

## Requirements

Requires [Torch7](http://torch.ch/), and uses CUDA if available. Also requires the following extra luarocks packages:

- luaposix
- moses
- logroll
- classic
- torchx
- rnn
- dpnn
- nninit
- tds
- **xitari**
- **alewrap**
- **rlenvs**

xitari, alewrap and rlenvs can be installed using the following commands:

```sh
luarocks install https://raw.githubusercontent.com/lake4790k/xitari/master/xitari-0-0.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/alewrap/master/alewrap-0-0.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/rlenvs/master/rocks/rlenvs-scm-1.rockspec
```

## Acknowledgements

- [@GeorgOstrovski](https://github.com/GeorgOstrovski) for confirmation on network usage in advantage operators + note on interaction with Double DQN.
- [@schaul](https://github.com/schaul) for clarifications on prioritised experience replay + dueling DQN hyperparameters.

## References

[1] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[2] [Increasing the Action Gap: New Operators for Reinforcement Learning](http://arxiv.org/abs/1512.04860)  
[3] [Deep Exploration via Bootstrapped DQN](http://arxiv.org/abs/1602.04621)  
[4] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[5] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[6] [Deep Recurrent Q-Learning for Partially Observable MDPs](http://arxiv.org/abs/1507.06527)  
[7] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[8] [The Arcade Learning Environment: An Evaluation Platform for General Agents](http://arxiv.org/abs/1207.4708)  
[9] [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](http://arxiv.org/abs/1312.6034)  
[10] [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806)  
[11] [Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901)  
[12] [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783)  
