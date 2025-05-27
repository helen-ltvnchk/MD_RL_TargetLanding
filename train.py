import gym
import numpy as np

# Импортируем нашу среду
from environment import DroneLandingEnv

def run_test(env, random_actions=True):
    observation = env.reset()
    total_reward = 0

    for step in range(env.max_steps):  # Максимальное количество шагов
        # Выбор действия
        if random_actions:
            action = env.action_space.sample()  # Случайное действие
        else:
            # Простая стратегия: пытаемся приблизиться к цели
            distance_to_target = observation[4]  # Расстояние до цели (5-й элемент наблюдения)
            if distance_to_target > 5:
                action = np.array([0.5, 0.5])  # Умеренная тяга вверх и вправо
            else:
                action = np.array([0.0, -0.5])  # Уменьшаем вертикальную тягу для посадки

        # Шаг в среде
        observation, reward, done, info = env.step(action)
        total_reward += reward

        # Визуализация
        env.render()

        # Проверка завершения эпизода
        if done:
            print(f"Эпизод завершен на шаге {step}. Общая награда: {total_reward:.2f}")
            break

    # Закрываем окно визуализации
    env.close()

if __name__ == "__main__":
    # Создаем экземпляр среды
    env = DroneLandingEnv()

    # Запускаем тест
    print("Тестируем среду с случайными действиями:")
    run_test(env, random_actions=True)

    print("\nТестируем среду с простой стратегией:")
    run_test(env, random_actions=False)