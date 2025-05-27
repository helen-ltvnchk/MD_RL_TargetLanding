from environment import DroneLandingEnv
from stable_baselines3.common.env_checker import check_env

env = DroneLandingEnv()
check_env(env, warn=True) # Проверяем, корректно ли работает среда
