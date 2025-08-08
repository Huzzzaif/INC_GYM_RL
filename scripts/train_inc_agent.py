from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from inc_env import INCForwardEnv  # import your custom env

# Create a vectorized version of the environment
env = make_vec_env(lambda: INCForwardEnv(N=5), n_envs=1)

# Create and train the DQN model
model = DQN(
    policy="MultiInputPolicy", 
    env=env, 
    verbose=1,
    tensorboard_log="./logs/inc_dqn"
)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("models/inc_dqn_model")
