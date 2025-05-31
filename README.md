# PERPPO
We modify the PPO algorithm from SB3, introducing PER term into the buffer and PPO loss.\\
The modification is defined in `PER.py`, where the `RolloutBuffer` and `PPO` class are inherited.\\
Install new version of SB3 (our code is based on `SB3==2.6.0`):
```sh
pip install stable_baselines3==2.6.0
```
Run the PERPPO algorithm:
```sh
bash pianomime/single_task/run_perppo.sh
```
Run the baseline PPO algorithm:
```sh
bash pianomime/single_task/run_ppo.sh
```