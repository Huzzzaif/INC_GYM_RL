"""
PLan
Load the graph from json file
Initialize the node att
Define packet structure 
provide the routing algos
"""
import json
import networkx as nx
import os
import numpy as np

# Constants (can be made configurable)
MAX_NEIGHBOURS = 10
MAX_PACKET_AGE_s = 10.0

def load_topology(json_path, seed=None):
    """
    Load a topology JSON and return a NetworkX graph with initialized attributes.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])

    rng = np.random.default_rng(seed)

    for node_id, pos in data["positions"].items():
        node = int(node_id)
        G.nodes[node]["pos"] = tuple(pos)

    for node_id, role in data.get("roles", {}).items():
        node = int(node_id)
        G.nodes[node]["role"] = role

    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        conn_degree = len(neighbors) / MAX_NEIGHBOURS

        G.nodes[node].update({
            "latency": rng.uniform(5, 300) / 1000.0,         # 5ms to 300ms → 0–0.3
            "battery": rng.uniform(40, 100) / 100.0,         # 40% to 100% → 0.4–1.0
            "trust": rng.uniform(0.4, 1.0),                  # 0.4–1.0
            "aggregation": rng.uniform(0, 1.0),              # 0–1
            "conn_degree": conn_degree,                     # normalized
            "encrypt_power": rng.uniform(0.5, 2.0) / 2.0,    # 0.25–1.0
            "buffer_load": rng.uniform(0, 1.0),              # 0–1
            "packet_age": rng.uniform(0, MAX_PACKET_AGE_s) / MAX_PACKET_AGE_s,
            "buffer": [],
            "active": True,
        })

    return G
class Packet:
    def __init__(self, source, destination, created_time):
        self.source = source
        self.destination = destination
        self.created_time = created_time  # Timestamp when packet was created
        self.path = [source]              # Track path for debugging
        self.age = 0                      # Time elapsed in network
        self.delivered = False            # Flag for delivery
        self.hops = 0                     # Number of hops
        self.encrypted = False           # Optional: track encryption state

    def update_path(self, node_id):
        self.path.append(node_id)
        self.hops += 1
    def __repr__(self):
        return f"<Packet src={self.source}, dst={self.destination}, hops={self.hops}, delivered={self.delivered}>"

# G = load_topology("network_sim/topologies/rgg_locked/rgg_N20_1.json", seed=42)
# # Sample features for node 7
# node7 = G.nodes[7]
# print({
#     "battery": node7["battery"],
#     "trust": node7["trust"],
#     "encrypt_power": node7["encrypt_power"],
#     "latency": node7["latency"]
# })