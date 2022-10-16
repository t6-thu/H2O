# H2O
<!-- ## When to Trust Your Simulator: Dynamics-AwareHybrid Offline-and-Online Reinforcement Learning -->
H2O ([https://arxiv.org/abs/2206.13464](https://arxiv.org/abs/2206.13464)) is the first Hybrid Offline-and-Online Reinforcement Learning framework, that enables simultaneous policy learning with offline real-world datasets and simulation rollouts, while also addressing the sim-to-real dynamics gaps in imperfect simulation. H2O introduces a dynamics-aware policy evaluation scheme, which adaptively penalizes the Q-values as well as fixes the Bellman error on simulated samples with large dynamics gaps. Through extensive simulation and real-world tasks, as well as theoretical analysis, we demonstrate the superior performance of H2O against other cross-domain online and offline RL algorithms. This repository provides the codebase on which we benchmark H2O and baselines in MuJoCo environments.

## Installation and Setups
To install the dependencies, run the command:
```python
    pip install -r requirements.txt
```
Add this repo directory to your `PYTHONPATH` environment variable:
```
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Run Benchmark Experiments
We benchmark H2O and its baselines on MuJoCo simulation environment and D4RL datasets. To begin, enter the folder `SimpleSAC`:
```
    cd SimpleSAC
```
Then you can run H2O experiments using the following example commands.
### Simulated in HalfCheetah-v2 with 2x gravity and Medium Replay dataset
```python
    python sim2real_sac_main.py \
        --env_list HalfCheetah-v2 \
        --data_source medium_replay \
        --unreal_dynamics gravity \
        --variety_list 2.0 
```
### Simulated in Walker-v2 with .3x friction and Medium Replay dataset
```python
    python sim2real_sac_main.py \
        --env_list Walker-v2 \
        --data_source medium_replay \
        --unreal_dynamics friction \
        --variety_list 0.3 
```
### Simulated in HalfCheetah-v2 with joint noise N(0,1) and Medium dataset
```python
    python sim2real_sac_main.py \
        --env_list HalfCheetah-v2 \
        --data_source medium \
        --variety_list 1.0 \
        --joint_noise_std 1.0 
```

## Visualization of Learning Curves
You can resort to [wandb](https://wandb.ai/site) to login your personal account with your wandb API key.
```
    export WANDB_API_KEY=YOUR_WANDB_API_KEY
```
and run `wandb online` to turn on the online syncronization.

## Citation
If you are using H2O framework or code for your project development, please cite the following paper:
```
@inproceedings{
  niu2022when,
  title={When to Trust Your Simulator: Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning},
  author={Haoyi Niu and Shubham Sharma and Yiwen Qiu and Ming Li and Guyue Zhou and Jianming HU and Xianyuan Zhan},
  booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=zXE8iFOZKw}
}
```
