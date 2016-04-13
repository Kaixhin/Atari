#!/bin/bash

# Specify paper/hyperparameters
if [ -z "$1" ]; then
  echo "Please enter paper, e.g. ./run nature"
  echo "Choices: nature|doubleq|duel|prioritised|priorduel|persistent|bootstrap"
  echo "Alternative choice: demo (for Catch)"
  exit 0
else
  PAPER=$1
  shift
fi

# Specify game
if [ "$PAPER" != "demo" ]; then
  if [ -z "$1" ]; then
    echo "Please enter game, e.g. ./run nature breakout"
    exit 0
  else
    GAME=$2
    shift
  fi
fi

if [ "$PAPER" == "demo" ]; then
  # Catch demo
  th main.lua -optimiser adam -steps 500000 -learnStart 10000 -tau 4 -memSize 10000 -epsilonSteps 10000 -valFreq 10000 -valSteps 6000 -bootstraps 0 -PALpha 0 "$@"
elif [ "$PAPER" == "nature" ]; then
  # Nature
  th main.lua -game $GAME -duel false -bootstraps 0 -memPriority none -epsilonEnd 0.1 -tau 10000 -doubleQ false -PALpha 0 -eta 0.00025 -gradClip 0 "$@"
elif [ "$PAPER" == "doubleq" ]; then
  # Double-Q (tuned)
  th main.lua -game $GAME -duel false -bootstraps 0 -memPriority none -PALpha 0 -eta 0.00025 -gradClip 0 "$@"
elif [ "$PAPER" == "duel" ]; then
  # Duel (eta is apparently lower but not specified in paper; unclear whether DDQN or tuned DDQN parameters are used)
  th main.lua -game $GAME -bootstraps 0 -memPriority none -PALpha 0 -eta 0.00025 "$@"
elif [ "$PAPER" == "prioritised" ]; then
  # Prioritised (rank-based)
  th main.lua -game $GAME -duel false -bootstraps 0 -alpha 0.7 -betaZero 0.5 -PALpha 0 -gradClip 0 "$@"
elif [ "$PAPER" == "priorduel" ]; then
  # Duel with rank-based prioritised experience replay (in duel paper)
  th main.lua -game $GAME -bootstraps 0 -alpha 0.7 -betaZero 0.5 -PALpha 0 "$@"
elif [ "$PAPER" == "persistent" ]; then
  # Persistent
  th main.lua -game $GAME -duel false -bootstraps 0 -memPriority none -epsilonEnd 0.1 -tau 10000 -doubleQ false -eta 0.00025 -gradClip 0 "$@"
elif [ "$PAPER" == "bootstrap" ]; then
  # Bootstrap
  th main.lua -game $GAME -duel false -memPriority none -tau 10000 -PALpha 0 -eta 0.00025 -gradClip 0 "$@"
fi
