#!/bin/bash

# Switch to script directory
cd `dirname -- "$0"`

# Specify paper/hyperparameters
if [ -z "$1" ]; then
  echo "Please enter paper, e.g. ./run nature"
  echo "Choices: nature|doubleq|duel|prioritised|priorduel|persistent|bootstrap|recurrent|async"
  echo "Alternative choice: demo|async-demo|async-demo-a3c (for Catch)"
  exit 0
else
  PAPER=$1
  shift
fi

# Specify game
if ! [[ "$PAPER" =~ demo ]]; then
  if [ -z "$1" ]; then
    echo "Please enter game, e.g. ./run nature breakout"
    exit 0
  else
    GAME=$1
    shift
  fi
fi

if [[ "$PAPER" =~ async ]]; then
  echo "Async mode specified"
  export OMP_NUM_THREADS=1 
fi

if [ "$PAPER" == "demo" ]; then
  # Catch demo
  th main.lua -gpu 0 -hiddenSize 32 -optimiser adam -steps 500000 -learnStart 20000 -tau 4 -memSize 20000 -epsilonSteps 10000 -valFreq 10000 -valSteps 6000 -bootstraps 0 -PALpha 0 "$@"
elif [ "$PAPER" == "nature" ]; then
  # Nature
  th main.lua -game $GAME -duel false -bootstraps 0 -memPriority none -epsilonEnd 0.1 -tau 10000 -doubleQ false -PALpha 0 -eta 0.00025 -gradClip 0 "$@"
elif [ "$PAPER" == "doubleq" ]; then
  # Double-Q (tuned)
  th main.lua -game $GAME -duel false -bootstraps 0 -memPriority none -PALpha 0 -eta 0.00025 -gradClip 0 "$@"
elif [ "$PAPER" == "duel" ]; then
  # Duel (eta is apparently lower but not specified in paper)
  # Note from Tom Schaul: Tuned DDQN hyperparameters are used
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
elif [ "$PAPER" == "recurrent" ]; then
  # Recurrent (note that evaluation methodology is different)
  th main.lua -game $GAME -histLen 10 -duel false -bootstraps 0 -recurrent true -memSize 400000 -memSampleFreq 1 -memPriority none -epsilonEnd 0.1 -tau 10000 -doubleQ false -PALpha 0 -optimiser adadelta -eta 0.1 "$@"

# Async modes
elif [ "$PAPER" == "async-demo" ]; then
  # NStepQ Catch demo
  th main.lua -async NStepQ -eta 0.00025 -momentum 0.99 -bootstraps 0 -batchSize 5 -hiddenSize 32 -doubleQ false -duel false -optimiser adam -steps 15000000 -tau 4 -memSize 20000 -epsilonSteps 10000 -valFreq 10000 -valSteps 6000 -bootstraps 0 -PALpha 0 "$@"
elif [ "$PAPER" == "async-demo-a3c" ]; then
  # A3C Catch demo
  th main.lua -async A3C -eta 0.0007 -momentum 0.99 -bootstraps 0 -batchSize 5 -hiddenSize 32 -doubleQ false -duel false -optimiser adam -steps 15000000 -tau 4 -memSize 20000 -epsilonSteps 10000 -valFreq 10000 -valSteps 6000 -bootstraps 0 -PALpha 0 "$@"
elif [ "$PAPER" == "async-nstep" ]; then
  # steps for "one day"  = 80 * 1e6
  # steps for "4 days" = 1e9
  th main.lua -async NStepQ -bootstraps 0 -batchSize 5 -momentum 0.99 -rmsEpsilon 0.1 -steps 80000000 -game $GAME -duel false -tau 40000 -optimiser sharedRmsProp -epsilonSteps 4000000 -doubleQ false -PALpha 0 -eta 0.0007 -gradClip 0 "$@"
elif [ "$PAPER" == "async-a3c" ]; then
  th main.lua -async A3C -bootstraps 0 -batchSize 5 -momentum 0.99 -rmsEpsilon 0.1 -steps 80000000 -game $GAME -duel false -tau 40000 -optimiser sharedRmsProp -epsilonSteps 4000000 -doubleQ false -PALpha 0 -eta 0.0007 -gradClip 0 "$@"
else
  echo "Invalid options"
fi
