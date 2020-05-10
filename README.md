# Pytorch Reinforcement Learning

This repository contains the code for policy gradient algorithm incorporating
with credit assignment mechanism.

## Install Dependencies

1. Install Pytorch 

```bash
pip install torch torchvision
```

2. install Tensorflow 2

```bash
pip install tensorflow=2.2
```
or 
```bash
pip install tensorflow-gpu=2.2
```

3. Install [OpenAI baseline](https://github.com/openai/baselines/tree/tf2) (Tensorflow 2 version)
```bash
git clone https://github.com/openai/baselines.git -b tf2 && \
cd baselines && \
pip install -e .
```

**Note**: I haven't tested the code on Tensorflow 1 yet but it should work as well.

4. Install gym
```bash
pip install 'gym[atari]'
```

5. Install [Park Platform](https://github.com/park-project/park). I modified the platform slightly to make it compatible with OpenAI's baseline.
```bash
git clone https://github.com/lehduong/park &&\
cd park && \
pip install -e .
```

## Run experiments
```bash 
python main.py --algo a2c --env-name PongNoFrameskip-v4
```
## Acknowledgement

The started code is based on [ikostrikov's repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
