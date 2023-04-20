import numpy as np
import os

def load_qtable(state_size, action_size):
    if os.path.exists("qtable.npy"):
        qtable = np.load("qtable.npy")
        print("Tabela Q carregada com sucesso!")
    else:
        qtable = np.zeros((state_size, action_size))
        print("Criando uma nova tabela Q!")
    return qtable

def save_qtable(qtable):
    np.save("qtable.npy", qtable)
    print("A tabela Q foi salva com sucesso!")

def test_qtable(qtable, env, num_episodes=100):
    total_rewards = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = 0
        while not done:
            action = np.argmax(qtable[state, :])
            new_state, reward, done, _ = env.step(action)
            episode_rewards += reward
            state = new_state
        total_rewards += episode_rewards
    return total_rewards / num_episodes
