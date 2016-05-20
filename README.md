# Atari ![Space Invader](http://www.rw-designer.com/cursor-view/74522.png)

**Work In Progress**

Asynchrounous version [[1]](#references) of DQN that runs on CPU multithreaded

Async modes supported

* 1 step Q
* 1 step Sarsa
* N step Q
* A3C

Q learning can be combined with these methods:

* double Q learning
* dueling
* ~~PAL~~ 
* ~~deep recurrent~~

This branch will be merged with the experience replay based `master` branch after refactoring unifying all the methods

## Requirements

Those listed on the `master` branch, but use this xitari instead with some modification for multithreaded usage:

```
luarocks install https://raw.githubusercontent.com/lake4790k/xitari/master/xitari-0-0.rockspe
```

And in addition install

- tds


## Acknowledgements

- [@Kaixhin](https://github.com/Kaixhin) for the experience replay based DQN implementation this branch is based on

## References

[1] [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783)  
