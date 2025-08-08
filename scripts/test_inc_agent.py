from stable_baselines3 import DQN
from inc_env import INCForwardEnv

# Load model
model = DQN.load("models/inc_dqn_model")

# Test environment
env = INCForwardEnv(N=5)
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    print(f"Action: {action}, Reward: {reward}")

print("Episode finished. Total reward:", total_reward)
