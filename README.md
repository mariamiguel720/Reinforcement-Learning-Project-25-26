## Project Overview

This project applies Reinforcement Learning to the **ICU-Sepsis** environment, which simulates sequential treatment decisions (vasopressor dose + IV fluid) for septic ICU patients. The goal is to train agents that maximise patient survival.

The work is divided into two configurations:

- **Config A** — discrete state space (714 states as integers), tabular methods: Q-Learning and Dyna-Q
- **Config B** — continuous state space (47 physiological features per state), deep RL: DQN, Double DQN, and PPO

## Project Structure

```
├── RL Config_A.ipynb          # Config A notebook — tabular methods (Q-Learning, Dyna-Q)
├── RL_Config_B.ipynb          # Config B notebook — deep RL methods (DQN, Double DQN, PPO)
├── Extra_Models.ipynb         # Additional experiments (SAC and other exploratory models)
│
├── envs/                      # Environment definitions
│   ├── env_setup.py           # Shared constants and base environment factory
│   ├── continuous_sepsis_env.py  # Continuous observation space wrapper (47 features)
│   └── wrappers.py            # Clinical wrappers: noisy obs, missing features, acute events
│
├── src/                       # Algorithm implementations and utilities
│   ├── functions_Config_A.py  # Q-Learning and Dyna-Q training functions (Config A)
│   ├── dqn_functions.py       # DQN, Double DQN, replay buffer, Optuna objective (Config B)
│   ├── ppo_functions.py       # PPO actor-critic, GAE, Optuna objective (Config B)
│   ├── utils_config_B.py      # Evaluation, rollout, and plotting utilities (Config B)
│   └── creative_extension.py  # Ablation study and SHAP analysis
│
├── models/                    # Saved model weights and training artefacts
│   ├── dqn_run1.pth / ppo_run1.pth        # Initial run checkpoints
│   ├── optuna_dqn/dqn_final/              # Best DQN model per seed after Optuna tuning
│   ├── double_dqn/                        # Double DQN checkpoints per seed
│   ├── optuna_ppo/ppo_final/              # Best PPO model per seed after Optuna tuning
│   ├── ablation/ablation_results.csv      # Ablation study results
│   └── sac/                               # SAC experimental models
│
└── figures/                   # Saved plots
    ├── config_A/              # Learning curves and results for Config A
    └── config_B/              # Learning curves and results for Config B
```


Each notebook runs end-to-end: environment exploration -> random baseline -> initial training -> Optuna hyperparameter tuning -> final retraining across 3 seeds -> comparative evaluation. Model checkpoints and results are saved to `models/` so training does not need to be repeated between sessions.
