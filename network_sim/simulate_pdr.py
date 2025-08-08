import os
import argparse
import numpy as np
from common import load_topology, Packet

from drift_final.rkl import evaluate_rkl
from baselineModels.shortest_path import evaluate_shortest_path
from baselineModels.random_forwarding import evaluate_random

def generate_packets(G, num_packets, cloud_id, creation_time=0):
    packets = []
    for _ in range(num_packets):
        source = np.random.choice([
            n for n in G.nodes if G.nodes[n].get("role") != "cloud"
        ])
        packets.append(Packet(source, cloud_id, created_time=creation_time))
    return packets

def run_simulation(json_path, num_packets=100, seed=42):
    # Load graph
    G = load_topology(json_path, seed=seed)

    # Identify cloud node
    cloud_id = next((n for n, d in G.nodes(data=True) if d.get("role") == "cloud"), None)
    if cloud_id is None:
        raise ValueError("No cloud node found in topology!")

    # Generate packets
    packets = generate_packets(G, num_packets, cloud_id)

    # Run all evaluations
    results = {
        "RKL": evaluate_rkl(G, packets),
        "ShortestPath": evaluate_shortest_path(G, packets),
        "Random": evaluate_random(G, packets)
    }

    # Print results
    print(f"\nResults for: {os.path.basename(json_path)}")
    for algo, res in results.items():
        print(f"{algo} → PDR: {res['pdr']:.2%} | Avg Hops: {res['avg_hops']:.2f} | Avg Latency: {res['avg_latency']:.3f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate PDR for one topology")
    parser.add_argument("--file", type=str, required=True, help="Path to JSON topology")
    parser.add_argument("--packets", type=int, default=100, help="Number of packets to send")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_simulation(args.file, args.packets, args.seed)
