import networkx as nx
import matplotlib.pyplot as plt
import json, random
from inc_env import make_env 
with open("episode_logs.json") as f:
    ep = json.load(f)[10]          # just pick the first episode for demo

N, seed, path = ep["N"], ep["seed"], ep["path"]

# recreate the same graph so positions match evaluation
env = make_env(N=N, graph_seed=seed)()
G = nx.Graph()
for node, nbrs in env.unwrapped.neighbors.items():
    for nbr in nbrs:
        G.add_edge(node, nbr)

pos = nx.spring_layout(G, seed=42)      # nice stable layout

# draw everything light-grey
nx.draw(G, pos, node_color="lightgrey", edge_color="lightgrey", with_labels=True)

# highlight the path in red
path_edges = list(zip(path[:-1], path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color="red")
nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="red")

plt.title(f"Agent path on N={N}, seed={seed}\\nReward={ep['reward_sum']:.2f}")
plt.tight_layout()
plt.show()
