#!/bin/bash

# Switch to script directory
cd `dirname -- "$0"`

# Specify paper/hyperparameters
if [ -z "$1" ]; then
  echo "Please enter paper and async method, e.g. ./async_run nature"
  echo "Paper choices: nature|doubleq|duel|persistent"
  echo "Async choices: 1stepq"
  echo "Alternative choice: demo (for Catch)"
  exit 0
else
  PAPER=$1
  shift
fi

ASYNC=1stepq  

# Specify game
if [ "$PAPER" != "demo" ]; then
  if [ -z "$1" ]; then
    echo "Please enter game, e.g. ./run nature breakout"
    exit 0
  else
    GAME=$1
    shift
  fi
fi

export OMP_NUM_THREADS=1 

if [ "$PAPER" == "demo" ]; then
  # Catch demo
  th async_main.lua -async $ASYNC -eta 0.00025 -doubleQ false -duel false -optimiser adam -steps 15000000 -tau 4 -epsilonSteps 10000 -valFreq 10000 -valSteps 6000 -PALpha 0 "$@"
elif [ "$PAPER" == "nature" ]; then
  # Nature
  th async_main.lua -async $ASYNC -game $GAME -duel false -tau 40000 -optimiser sharedRmsProp -epsilonSteps 4000000 -doubleQ false -PALpha 0 -eta 0.0007 -gradClip 0 "$@"
elif [ "$PAPER" == "doubleq" ]; then
  # Double-Q (tuned)
  th async_main.lua -async $ASYNC -game $GAME -duel false -tau 40000 -optimiser sharedRmsProp -epsilonSteps 4000000 -PALpha 0 -eta 0.0007 -gradClip 0 "$@"
elif [ "$PAPER" == "duel" ]; then
  # Duel (eta is apparently lower but not specified in paper; unclear whether DDQN or tuned DDQN parameters are used)
  th async_main.lua -async $ASYNC -game $GAME -PALpha 0 -eta 0.00025 "$@"
elif [ "$PAPER" == "persistent" ]; then
  # Persistent
  th async_main.lua -async $ASYNC -game $GAME -duel false -epsilonEnd 0.1 -tau 10000 -doubleQ false -eta 0.00025 -gradClip 0 "$@"
fi
