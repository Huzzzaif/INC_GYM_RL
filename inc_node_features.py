import random
import pandas as pd

def generate_node_features(N):
    def generate_single_node(node_id):
        return {
            "Node ID": f"N{node_id+1}",
            "Battery Level (%)": round(random.uniform(10, 100), 2),
            "Connectivity Degree": random.randint(1, N-1),
            "Aggregation Potential": random.randint(0, 5),
            "Current Latency (ms)": round(random.uniform(5, 100), 2),
            "Encryption Power (MBps)": round(random.uniform(1, 20), 2),
            "Load Factor (%)": round(random.uniform(0, 100), 2),
            "Trust Score (0–1)": round(random.uniform(0.4, 1.0), 2),
            "Packet Age (s)": round(random.uniform(0, 10), 2)
        }
    
    return [generate_single_node(i) for i in range(N)]

def get_feature_dataframe(N=5):
    node_feature_list = generate_node_features(N)
    return pd.DataFrame(node_feature_list)
def get_node_ids_and_features(N=5):
    df = get_feature_dataframe(N)
    node_ids = df["Node ID"].tolist()
    features = df.drop(columns=["Node ID"]).to_dict(orient="index")
    return node_ids, features
