from stable_baselines3 import PPO
from environment import DroneLandingEnv
from stable_baselines3.common.monitor import Monitor

env = Monitor(DroneLandingEnv(), filename="logs/") # Инициализируем экземпляр среды

#model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[128, 128])) # создаём объект модели - PPO-агент
model = PPO.load("ppo_drone_landing", env=env)
model.learn(total_timesteps=300000) # запускаем обучение на 50000 шагов (кол-во шагов можно уменьшать или увеличивать)

model.learn(total_timesteps=100000, reset_num_timesteps=False)
model.save("ppo_drone_landing") # Сохраняем модель для дальнейшей работы и тестирования