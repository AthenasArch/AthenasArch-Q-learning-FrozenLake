#   *******************************************************
#
#                       V1 OK
# 
#   *******************************************************

import numpy as np
import gym
import random

env = gym.make("FrozenLake-v1")
action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 15000
learning_rate = 0.8
max_steps = 99
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

rewards = []

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward
        state = new_state

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)

env.reset()
rewards = []

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)
        if done:
            env.render()
            rewards.append(reward)
            print("Number of steps", step)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / 5))


# o objetivo do jogo FrozenLake é encontrar o caminho até o
#  objetivo (G) evitando buracos (H) e acumulando a maior pontuação 
# possível. A cada episódio, o jogo é reiniciado.
#   *******************************************************
#
#                       V2 OK
# 
#   *******************************************************


# import gym
# import numpy as np
# import pygame
# import time

# pygame.init()

# grid_size = 4
# cell_size = 125

# width = grid_size * cell_size
# height = grid_size * cell_size

# screen = pygame.display.set_mode((width, height))
# pygame.display.set_caption("FrozenLake")

# lake_image = pygame.image.load("frozenlake.png")

# def draw_grid(env, path):
#     for i in range(grid_size):
#         for j in range(grid_size):
#             cell = env.desc[i][j].decode("utf-8")

#             if cell == "S":
#                 cell_image = pygame.transform.scale(lake_image.subsurface((0, 0, 250, 250)), (cell_size, cell_size))
#             elif cell == "F":
#                 cell_image = pygame.transform.scale(lake_image.subsurface((250, 0, 250, 250)), (cell_size, cell_size))
#             elif cell == "H":
#                 cell_image = pygame.transform.scale(lake_image.subsurface((0, 250, 250, 250)), (cell_size, cell_size))
#             elif cell == "G":
#                 cell_image = pygame.transform.scale(lake_image.subsurface((250, 250, 250, 250)), (cell_size, cell_size))

#             screen.blit(cell_image, (j * cell_size, i * cell_size))

#             if (i, j) in path:
#                 pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size), 4)

# def draw_env(env, q_table, episode, path):
#     screen.fill((255, 255, 255))
#     draw_grid(env, path)

#     pygame.display.flip()
#     time.sleep(0.5)

# def q_learning(env, num_episodes=2000, learning_rate=0.8, gamma=0.95, epsilon=1, epsilon_decay=0.999, min_epsilon=0.1):
#     q_table = np.zeros([env.observation_space.n, env.action_space.n])

#     for episode in range(num_episodes):
#         state = env.reset()
#         done = False
#         step = 0

#         path = []

#         while not done:
#             if np.random.uniform(0, 1) < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = np.argmax(q_table[state, :])

#             next_state, reward, done, _ = env.step(action)

#             x, y = state % grid_size, state // grid_size
#             path.append((y, x))

#             q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
#             state = next_state

#             step += 1

#         epsilon = max(min_epsilon, epsilon * epsilon_decay)

#         draw_env(env, q_table, episode, path)

#     return q_table

# def main():
#     env = gym.make("FrozenLake-v1", is_slippery=False)
#     q_table = q_learning(env)
#     print("Q-table:")
#     print(q_table)

# if __name__ == "__main__":
#     main()




# o objetivo do jogo FrozenLake é encontrar o caminho até o
#  objetivo (G) evitando buracos (H) e acumulando a maior pontuação 
# possível. A cada episódio, o jogo é reiniciado.
#   *******************************************************
#
#                       V3 OK
# 
#   *******************************************************
# import gym
# import numpy as np
# import pygame
# import time

# pygame.init()
# screen = pygame.display.set_mode((500, 600))
# pygame.display.set_caption("FrozenLake-v1 Visualization")

# env = gym.make("FrozenLake-v1")
# state_size = env.observation_space.n
# action_size = env.action_space.n

# qtable = np.zeros((state_size, action_size))

# frozenlake_img = pygame.image.load("frozenlake.png")
# frozenlake_img = pygame.transform.scale(frozenlake_img, (500, 500))


# def draw_path(path, episode_reward, episode_number):
#     screen.fill((255, 255, 255))
#     screen.blit(frozenlake_img, (0, 0))

#     for i, s in enumerate(path):
#         x = (s % 4) * 125
#         y = (s // 4) * 125

#         pygame.draw.rect(screen, (255, 0, 0), (x, y, 125, 125), 5)

#     font = pygame.font.Font(None, 36)
#     text = font.render(f"Episode: {episode_number}", True, (0, 0, 0))
#     screen.blit(text, (10, 510))

#     text = font.render(f"Reward: {episode_reward}", True, (0, 0, 0))
#     screen.blit(text, (10, 540))

#     pygame.display.flip()
#     time.sleep(0.2)

# def test_agent(qtable):
#     state = env.reset()
#     done = False
#     path = []
#     total_rewards = 0

#     while not done:
#         action = np.argmax(qtable[state, :])
#         new_state, reward, done, _ = env.step(action)
#         path.append(new_state)
#         total_rewards += reward
#         state = new_state

#     return path, total_rewards


# def main():
#     # total_episodes = 10000
#     total_episodes = 3000
#     learning_rate = 0.8
#     max_steps = 99
#     gamma = 0.95
#     epsilon = 1.0
#     max_epsilon = 1.0
#     min_epsilon = 0.01
#     decay_rate = 0.01

#     rewards = []

#     for episode in range(total_episodes):
#         state = env.reset()
#         step = 0
#         done = False
#         total_rewards = 0
#         path = []

#         for step in range(max_steps):
#             exp_exp_tradeoff = np.random.uniform(0, 1)

#             if exp_exp_tradeoff > epsilon:
#                 action = np.argmax(qtable[state, :])
#             else:
#                 action = env.action_space.sample()

#             new_state, reward, done, info = env.step(action)
#             path.append(new_state)

#             qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

#             total_rewards += reward
#             state = new_state

#             if done:
#                 break

#         epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
#         rewards.append(total_rewards)

#         if episode % 10 == 0:
#             draw_path(path, total_rewards, episode)

#     print("Score over time: " + str(sum(rewards) / total_episodes))
#     print(qtable)


# if __name__ == "__main__":
#     main()
#     path, total_rewards = test_agent(qtable)
#     draw_path(path, total_rewards, "Test")
#     time.sleep(20)
















# o objetivo do jogo FrozenLake é encontrar o caminho até o
#  objetivo (G) evitando buracos (H) e acumulando a maior pontuação 
# possível. A cada episódio, o jogo é reiniciado.
#   *******************************************************
#
#                       V4 OK
# 
#   *******************************************************
# import gym
# import numpy as np
# import pygame
# import time
# import matplotlib.pyplot as plt
# import sys

# rewards = []

# pygame.init()
# screen = pygame.display.set_mode((1000, 600))
# pygame.display.set_caption("FrozenLake-v1 Visualization")

# env = gym.make("FrozenLake-v1")
# state_size = env.observation_space.n
# action_size = env.action_space.n

# qtable = np.zeros((state_size, action_size))

# frozenlake_img = pygame.image.load("frozenlake.png")
# frozenlake_img = pygame.transform.scale(frozenlake_img, (500, 500))

# def draw_path(path, episode_reward, episode_number, rewards):
#     screen.fill((255, 255, 255))
#     screen.blit(frozenlake_img, (0, 0))

#     for i, s in enumerate(path):
#         x = (s % 4) * 125
#         y = (s // 4) * 125
#         pygame.draw.rect(screen, (255, 0, 0), (x, y, 125, 125), 5)

#     font = pygame.font.Font(None, 36)
#     text = font.render(f"Episode: {episode_number}", True, (0, 0, 0))
#     screen.blit(text, (10, 510))

#     text = font.render(f"Reward: {episode_reward}", True, (0, 0, 0))
#     screen.blit(text, (10, 540))

#     plot_rewards(rewards)

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()

#     pygame.display.flip()
#     # time.sleep(0.2)

# def plot_rewards(rewards):
#     cumulative_rewards = np.cumsum(rewards)
#     plt.plot(cumulative_rewards)
#     plt.xlabel("Episodes")
#     plt.ylabel("Cumulative Rewards")
#     plt.title("Cumulative Rewards vs Episode")
#     plt.tight_layout()
#     plt.savefig("cumulative_rewards_vs_episode.png")
#     plt.close()

#     graph = pygame.image.load("cumulative_rewards_vs_episode.png")
#     graph = pygame.transform.scale(graph, (500, 600))
#     screen.blit(graph, (500, 0))
#     pygame.display.flip()




# def draw_graph(rewards):
#     plt.plot(rewards)
#     plt.xlabel("Episodes")
#     plt.ylabel("Rewards")
#     plt.title("Reward vs Episode")
#     plt.tight_layout()
#     plt.savefig("reward_vs_episode.png")
#     plt.close()

#     graph = pygame.image.load("reward_vs_episode.png")
#     graph = pygame.transform.scale(graph, (500, 600))
#     screen.blit(graph, (500, 0))
#     pygame.display.flip()


# def test_agent(qtable):
#     state = env.reset()
#     done = False
#     path = []
#     total_rewards = 0

#     while not done:
#         action = np.argmax(qtable[state, :])
#         new_state, reward, done, _ = env.step(action)
#         path.append(new_state)
#         total_rewards += reward
#         state = new_state

#     return path, total_rewards

# def main():
#     total_episodes = 1000
#     learning_rate = 0.8
#     max_steps = 99
#     gamma = 0.95
#     epsilon = 1.0
#     max_epsilon = 1.0
#     min_epsilon = 0.01
#     decay_rate = 0.01

#     global rewards

#     for episode in range(total_episodes):
#         state = env.reset()
#         step = 0
#         done = False
#         total_rewards = 0
#         path = []

#         for step in range(max_steps):
#             exp_exp_tradeoff = np.random.uniform(0, 1)

#             if exp_exp_tradeoff > epsilon:
#                 action = np.argmax(qtable[state, :])
#             else:
#                 action = env.action_space.sample()

#             new_state, reward, done, info = env.step(action)
#             path.append(new_state)

#             qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

#             total_rewards += reward
#             state = new_state

#             if done:
#                 break

#         epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
#         rewards.append(total_rewards)

#         # draw_path(path, total_rewards, episode, rewards[:episode+1])  # Chame a função draw_path a cada iteração do loop
#         # para nao ficar chamando o grafico o tempo todo
#         if episode % 10 == 0:
#             draw_path(path, total_rewards, episode, rewards[:episode+1])


#     draw_graph(rewards)

#     print("Score over time: " + str(sum(rewards) / total_episodes))
#     print(qtable)

# if __name__ == "__main__":
#     main()
#     path, total_rewards = test_agent(qtable)
#     draw_path(path, total_rewards, "Test", rewards)
#     time.sleep(20)








# o objetivo do jogo FrozenLake é encontrar o caminho até o
#  objetivo (G) evitando buracos (H) e acumulando a maior pontuação 
# possível. A cada episódio, o jogo é reiniciado.
#   *******************************************************
#
#                       V5 OK - Salvando historico da tabela
# 
#   *******************************************************
# import gym
# import numpy as np
# import pygame
# import time
# import matplotlib.pyplot as plt
# import sys
# from qtable_manager import load_qtable, save_qtable, test_qtable

# rewards = []

# pygame.init()
# screen = pygame.display.set_mode((1000, 600))
# pygame.display.set_caption("FrozenLake-v1 Visualization")

# env = gym.make("FrozenLake-v1")
# state_size = env.observation_space.n
# action_size = env.action_space.n

# qtable = np.zeros((state_size, action_size))

# frozenlake_img = pygame.image.load("frozenlake.png")
# frozenlake_img = pygame.transform.scale(frozenlake_img, (500, 500))

# def draw_path(path, episode_reward, episode_number, rewards):
#     screen.fill((255, 255, 255))
#     screen.blit(frozenlake_img, (0, 0))

#     for i, s in enumerate(path):
#         x = (s % 4) * 125
#         y = (s // 4) * 125
#         pygame.draw.rect(screen, (255, 0, 0), (x, y, 125, 125), 5)

#     font = pygame.font.Font(None, 36)
#     text = font.render(f"Episode: {episode_number}", True, (0, 0, 0))
#     screen.blit(text, (10, 510))

#     text = font.render(f"Reward: {episode_reward}", True, (0, 0, 0))
#     screen.blit(text, (10, 540))

#     plot_rewards(rewards)

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()

#     pygame.display.flip()
#     # time.sleep(0.2)

# def plot_rewards(rewards):
#     cumulative_rewards = np.cumsum(rewards)
#     plt.plot(cumulative_rewards)
#     plt.xlabel("Episodes")
#     plt.ylabel("Cumulative Rewards")
#     plt.title("Cumulative Rewards vs Episode")
#     plt.tight_layout()
#     plt.savefig("cumulative_rewards_vs_episode.png")
#     plt.close()

#     graph = pygame.image.load("cumulative_rewards_vs_episode.png")
#     graph = pygame.transform.scale(graph, (500, 600))
#     screen.blit(graph, (500, 0))
#     pygame.display.flip()

# def draw_graph(rewards):
#     plt.plot(rewards)
#     plt.xlabel("Episodes")
#     plt.ylabel("Rewards")
#     plt.title("Reward vs Episode")
#     plt.tight_layout()
#     plt.savefig("reward_vs_episode.png")
#     plt.close()

#     graph = pygame.image.load("reward_vs_episode.png")
#     graph = pygame.transform.scale(graph, (500, 600))
#     screen.blit(graph, (500, 0))
#     pygame.display.flip()


# def test_agent(qtable):
#     state = env.reset()
#     done = False
#     path = []
#     total_rewards = 0

#     while not done:
#         action = np.argmax(qtable[state, :])
#         new_state, reward, done, _ = env.step(action)
#         path.append(new_state)
#         total_rewards += reward
#         state = new_state

#     return path, total_rewards

# def end_time(time_limit):
#     running = True
#     start_time = pygame.time.get_ticks()

#     while running:
#         current_time = pygame.time.get_ticks()
#         if current_time - start_time >= time_limit:
#             break

#         for event in pygame.event.get():
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     running = False
#             elif event.type == pygame.QUIT:
#                 running = False
#         pygame.time.delay(100)

# def main():
#     total_episodes = 1000
#     learning_rate = 0.8
#     max_steps = 99
#     gamma = 0.95
#     epsilon = 1.0
#     max_epsilon = 1.0
#     min_epsilon = 0.01
#     decay_rate = 0.01

#     global rewards

#     # Carregue a tabela Q ou crie uma nova
#     qtable = load_qtable(state_size, action_size)

#     # Teste a tabela Q carregada ou criada
#     initial_average_reward = test_qtable(qtable, env)
#     print(f"Recompensa média inicial: {initial_average_reward}")

#     for episode in range(total_episodes):
#         state = env.reset()
#         step = 0
#         done = False
#         total_rewards = 0
#         path = []

#         for step in range(max_steps):
#             exp_exp_tradeoff = np.random.uniform(0, 1)

#             if exp_exp_tradeoff > epsilon:
#                 action = np.argmax(qtable[state, :])
#             else:
#                 action = env.action_space.sample()

#             new_state, reward, done, info = env.step(action)
#             path.append(new_state)

#             qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

#             total_rewards += reward
#             state = new_state

#             if done:
#                 break

#         epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
#         rewards.append(total_rewards)

#         if episode % 10 == 0:
#             draw_path(path, total_rewards, episode, rewards[:episode+1])

#     draw_graph(rewards)

#     print("Score over time: " + str(sum(rewards) / total_episodes))
#     print(qtable)

#     # Teste a tabela Q após o treinamento
#     final_average_reward = test_qtable(qtable, env)
#     print(f"Recompensa média final: {final_average_reward}")

#     # Verifique se a nova tabela Q gerou melhores resultados
#     if final_average_reward > initial_average_reward:
#         # Salve a nova tabela Q e informe ao usuário
#         save_qtable(qtable)
#         print("A tabela Q foi atualizada com sucesso!")
#     else:
#         print("Não houve evolução no aprendizado.")


# if __name__ == "__main__":
#     main()
#     path, total_rewards = test_agent(qtable)
#     draw_path(path, total_rewards, "Test", rewards)
#     end_time(20000) 
