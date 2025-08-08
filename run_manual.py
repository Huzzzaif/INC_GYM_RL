from inc_env import INCForwardEnv
import numpy as np

env = INCForwardEnv()
obs, _ = env.reset()
done = False
step_num = 0
node_names = env.nodes

while not done:
    print(f"\n--- Step {step_num} ---")
    
    current_node_idx = obs['current_node_idx']
    current_node_name = node_names[current_node_idx]
    
    neighbor_dict = env.neighbors[current_node_name]
    neighbor_names = list(neighbor_dict.keys())
    neighbor_scores = obs['node_features'][:, 6]

    print(f"Current Node: {current_node_name}")
    print("Neighbor Scores:")

    for i, name in enumerate(neighbor_names):
        print(f"  {i}. {name} - Score: {neighbor_scores[i]}")

    # Select the neighbor with the highest score
    action = int(np.argmax(neighbor_scores))
    selected_neighbor = neighbor_names[action]
    selected_score = neighbor_scores[action]

    print(f"Agent selected: {selected_neighbor} with score {selected_score}")

    # Take the step
    obs, reward, done, _, _ = env.step(action)
    print(f"Reward: {reward}")
    step_num += 1

print("\nEpisode finished!")
