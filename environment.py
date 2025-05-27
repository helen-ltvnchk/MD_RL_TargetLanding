import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class DroneLandingEnv(gym.Env): # Создаём класс среды, наследуемся от gym.Env
    def __init__(self):
        super(DroneLandingEnv, self).__init__()

        # Вводим параметры среды
        self.gravity = 9.8 # Ускорение свободного падения
        self.engine_power = 15.0 # Максимальная тяга двигателей
        self.target_pos = np.array([0.0, 0.0]) # Целевая точка посадки (x=0, y=0)
        self.max_steps = 500 # Максимальное количество шагов в эпизоде
        self.current_step = 0 # текущий шаг

        # Пространство действий и пространство состояний
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        # Box - непрерывное пространство

        # Параметры для визуализации
        self.fig = None
        self.ax = None # fig и ax - Переменные для хранения объектов графика
        self.drone_trajectory = [] #  Список для хранения координат

    def reset(self, seed=None, options=None): # Сброс среды
        super().reset(seed=seed)

        self.drone_pos = np.array([np.random.uniform(-10, 10), 20.0]) # Инициализируемся в случайной точке в диапазоне (-10, 10) на высоте 20
        self.drone_vel = np.zeros(2) # Начальная скорость дрона 0
        self.fuel = 100.0 # Начальное значение  топлива
        self.current_step = 0 # Сброс счётчика шагов
        self.drone_trajectory = [self.drone_pos.copy()] # Сохраняем начальную позицию дрона
        return self._get_obs(), {}
    
    def step(self, action): # Обработка действия агента за 1 шаг
        thrust_vertical = np.clip(action[0], -1, 1) * self.engine_power # Вертикальная тяга
        thrust_horizontal = np.clip(action[1], -1, 1) * self.engine_power # Горизонтальная тяга

        # Ускорение
        acceleration = np.array([thrust_horizontal, thrust_vertical]) / 10.0 # Вычисляем ускорение дрона
        acceleration[1] -= self.gravity # Вычитаем гравитацию из вертикального ускорения

        self.drone_vel += acceleration * 0.1 # Обновляем скорость дрона
        self.drone_pos += self.drone_vel * 0.1 # Обновляем позицию дрона

        # Расход топлива
        fuel_consumption = np.abs(thrust_vertical) + np.abs(thrust_horizontal)
        self.fuel -= fuel_consumption * 0.01
        self.fuel = np.clip(self.fuel, 0, 100)

        self.current_step += 1
        self.drone_trajectory.append(self.drone_pos.copy())

        # Рассчёт награды
        reward = self._calculate_reward() # Вычисляем награду через вспомогательный метод

        terminated = False
        truncated = False

        # Проверка успешности завершения эпизода
        if self._check_landing(): # Если мы успешны
            terminated = True # Закончили шаг
            reward += 100
        elif self.drone_pos[1] < 0 or self.fuel <= 0:  # Если что-то пошло не так
            truncated = True 
            reward -= 100
        elif self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}
    
    # Визуализация
    def render(self, mode='human'): # mode - режим визуализации среды
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()

        self.ax.cla()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-5, 25)
        self.ax.scatter(*self.target_pos, c='red', s=100, label='Цель') # Отображается целевая точка

        # Отображение траектории дрона и его текущей позиции
        trajectory = np.array(self.drone_trajectory)
        self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'b--', alpha=0.5)
        self.ax.scatter(*self.drone_pos, c='blue', s=50, label='Дрон')
        
        # Добавление текста с информацией о топливе и скорости и обновление графика
        self.ax.text(
            -18, 22, 
            f'Топливо: {self.fuel:.1f}\n'
            f'Скорость: {np.linalg.norm(self.drone_vel):.1f} м/с',
            fontsize=10
        )
        self.ax.legend()
        plt.pause(0.01)

# Вспомогательные методы
    def _get_obs(self): # Формирование наблюдения
        distance = np.linalg.norm(self.drone_pos - self.target_pos)
        return np.array([
            self.drone_pos[0], self.drone_pos[1],
            self.drone_vel[0], self.drone_vel[1],
            distance, self.fuel
        ]).astype(np.float32)
    
    # Штраф за расстояние до цели, высокую скорость и расходтоплива
    # def _calculate_reward(self): # Вычисление награды
        # distance = np.linalg.norm(self.drone_pos - self.target_pos)
        # speed = np.linalg.norm(self.drone_vel)
        # return -distance * 0.1 - speed * 0.05 - (100 - self.fuel) * 0.01

        # reward = -distance * 0.1 - speed * 0.05

        # if self.drone_pos[1] < 1.0:
            # reward += 1.0
        # return reward
        #    return -distance * 0.1 - speed  * 0.5
        #else:
        #    return -distance * 0.1 - speed * 0.05

    def _calculate_reward(self):
    # Расстояние до цели и скорость
        distance = np.linalg.norm(self.drone_pos - self.target_pos)
        speed = np.linalg.norm(self.drone_vel)

    # Бонус за приближение к земле (стимулирует опускание)
        ground_bonus = max(0, (20 - self.drone_pos[1])) * 0.1  # ниже = лучше

    # Штраф за высокую скорость (мягкость посадки)
        speed_penalty = speed * 0.1

    # Бонус за близость к цели
        proximity_bonus = max(0, (10 - distance)) * 0.2

        soft_landing_bonus = 5.0 if self.drone_pos[1] < 1.0 else 0.0

    # Общая награда
        reward = ground_bonus + proximity_bonus - speed_penalty + soft_landing_bonus

        return reward

    
    def _check_landing(self): # Проверка успешной посадки
        #distance = np.linalg.norm(self.drone_pos - self.target_pos)
        #speed = np.linalg.norm(self.drone_vel)
        #return (distance < 2.0) and (speed < 1.0)
        return(self.drone_pos[1] < 1.0) and (np.linalg.norm(self.drone_vel) < 3.0) # Второй этап обучения
        #return self.drone_pos[1] < 1.0 # Первый этап обучения 
            

    def close(self):
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None

