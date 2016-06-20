# Atari ![Space Invader](http://www.rw-designer.com/cursor-view/74522.png)
[![Build Status](https://img.shields.io/travis/Kaixhin/Atari.svg)](https://travis-ci.org/Kaixhin/Atari)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/Kaixhin/Atari?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

**Work In Progress:** Crossed out items have been partially implemented.

~~Prioritised experience replay~~ [[1]](#references) persistent advantage learning [[2]](#references) ~~bootstrapped~~ [[3]](#references) dueling [[4]](#references) double [[5]](#references) deep ~~recurrent~~ [[6]](#references) Q-network [[7]](#references) for the Arcade Learning Environment [[8]](#references) (and [custom environments](#custom)). Or PERPALB(triple-D)RQN for short...

Additional asynchronous agents [[9]](#references):

- One-step Sarsa
- One-step Q-learning
- N-step Q-learning
- Advantage actor-critic

Run `th main.lua` to run headless, or `qlua main.lua` to display the game. The main options are `-game` to choose the ROM (see the [ROM directory](roms/README.md) for more details) and `-mode` as either `train` or `eval`. Can visualise saliency maps [[10]](#references), optionally using guided [[11]](#references) or "deconvnet" [[12]](#references) backpropagation. Saliency map modes are applied at runtime so that they can be applied retrospectively to saved models.

To run experiments based on hyperparameters specified in the individual papers, use `./run.sh <paper> <game> <args>`. `<args>` can be used to overwrite arguments specified earlier (in the script); for more details see the script itself. By default the code trains on a demo environment called Catch - use `./run.sh demo` to run the demo with good default parameters. Note that this code uses CUDA by default if available, but the Catch network is small enough that it runs faster on CPU.

In training mode if you want to quit using `Ctrl+C` then this will be caught and you will be asked if you would like to save the agent. Note that for non-asynchronous agents the experience replay memory will be included, totalling ~7GB. The main script also automatically saves the weights of the best performing DQN (according to the average validation score).

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

## Custom

You can use a custom environment (as the path to a Lua file/`rlenvs`-namespaced environment) using `-env`, as long as the class returned respects the `rlenvs` [API](https://github.com/Kaixhin/rlenvs#api). One restriction is that the state must be represented as a single tensor (with arbitrary dimensionality), and only a single discrete action must be returned. To prevent massive memory consumption for agents that use experience replay memory, states are discretised to integers ∈ [0, 255], assuming the state is comprised of reals ∈ [0, 1] - this can be disabled with `-discretiseMem false`. Visual environments can make use of explicit `-height`, `-width` and `-colorSpace` options to perform preprocessing for the network.

If the environment has separate behaviour during training and testing it should also implement `training` and `evaluate` methods - otherwise these will be added as empty methods during runtime. The environment can also implement a `getDisplay` method (with a mandatory `getDisplaySpec` method for determining screen size) which will be used for displaying the screen/computing saliency maps, where `getDisplay` must return a RGB (3D) tensor; this can also be utilised even if the state is not an image (although saliency can only be computed for states that are images). This **must** be implemented to have a visual display/computing saliency maps. The `-zoom` factor can be used to increase the size of small displays.

You can also use a custom model (body) with `-modelBody`, which replaces the usual DQN convolutional layers with a custom Torch neural network (as the path to a Lua file). The model will receive a stack of the previous states (as determined by `-histLen`), and must reshape them manually if needed. The DQN "heads" will then be constructed as normal, with `-hiddenSize` used to change the size of the fully connected layer if needed.

For an example on a GridWorld environment, run `./run.sh demo-grid` - the demo also works with `qlua` and experience replay agents. The custom environment and network can be found in the [examples](https://github.com/Kaixhin/Atari/tree/master/examples) folder.

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
[9] [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783)  
[10] [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](http://arxiv.org/abs/1312.6034)  
[11] [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806)  
[12] [Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901)  
