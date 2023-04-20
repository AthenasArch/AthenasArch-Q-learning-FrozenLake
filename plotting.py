import pygame
import matplotlib.pyplot as plt
import numpy as np

def draw_path(path, episode_reward, episode_number, rewards):
    screen.fill((255, 255, 255))
    screen.blit(frozenlake_img, (0, 0))

    for i, s in enumerate(path):
        x = (s % 4) * 125
        y = (s // 4) * 125
        pygame.draw.rect(screen, (255, 0, 0), (x, y, 125, 125), 5)

    font = pygame.font.Font(None, 36)
    text = font.render(f"Episode: {episode_number}", True, (0, 0, 0))
    screen.blit(text, (10, 510))

    text = font.render(f"Reward: {episode_reward}", True, (0, 0, 0))
    screen.blit(text, (10, 540))

    plot_rewards(rewards)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.flip()
    # time.sleep(0.2)

def plot_rewards(rewards):
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(cumulative_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.title("Cumulative Rewards vs Episode")
    plt.tight_layout()
    plt.savefig("cumulative_rewards_vs_episode.png")
    plt.close()

    graph = pygame.image.load("cumulative_rewards_vs_episode.png")
    graph = pygame.transform.scale(graph, (500, 600))
    screen.blit(graph, (500, 0))
    pygame.display.flip()

def draw_graph(rewards):
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Reward vs Episode")
    plt.tight_layout()
    plt.savefig("reward_vs_episode.png")
    plt.close()

    graph = pygame.image.load("reward_vs_episode.png")
    graph = pygame.transform.scale(graph, (500, 600))
    screen.blit(graph, (500, 0))
    pygame.display.flip()
