import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("rl_results.csv")  # Replace with your actual file path

# Group by baseline
baselines = df['baseline'].unique()

plt.figure(figsize=(10, 6))

# Plot one line per baseline
for b in baselines:
    subset = df[df['baseline'] == b]
    subset = subset.groupby('N').mean().reset_index()  # Average over different seeds
    plt.plot(subset['N'], subset['pdr'], marker='o', label=b)

plt.title("PDR vs Number of Nodes")
plt.xlabel("Number of Nodes (N)")
plt.ylabel("Packet Delivery Ratio (PDR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
