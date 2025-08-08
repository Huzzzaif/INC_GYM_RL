# DRIFT-RL: Dynamic Routing via Intelligent Forwarding using Trust-aware Reinforcement Learning

DRIFT-RL is a reinforcement learning–driven dynamic routing framework for In-Network Computing (INC) in privacy-preserving IoT networks.  
It combines **action-masked** dynamic Gymnasium environments, **Maskable-PPO** curriculum training, and multiple **non-RL baselines** for comparison.

The system is designed for research workflows, producing evaluation CSVs ready for plots and paper integration.

---

## 📌 Features
- **Custom Gymnasium environment** with:
  - Dynamic wireless topology generation
  - Node features (latency, energy, trust, aggregation potential, etc.)
  - Action masking to avoid invalid moves
- **Maskable-PPO training loop** with curriculum learning
- **Three baseline algorithms** for comparison:
  - EEHE
  - LEACH-C-HE
  - Q-Routing
- **Evaluation harness**:
  - Aggregates metrics like Packet Delivery Ratio (PDR), latency, and energy
  - Outputs results to CSV and JSON for analysis
- **Checkpointing** every 100 epochs


---

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/Huzzzaif/INC_GYM_RL.git
cd INC_GYM_RL
```
### Clone the repository
```bash
python3 -m venv incvenv
source incvenv/bin/activate  # Mac/Linux
incvenv\Scripts\activate     # Windows
```
### Install Requirements
```bash
pip install -r requirements.txt
```
### Train DRIFT-RL
```bash
python inc_env.py train --epochs 800 --out drift_final.zip
```
### Evaluate a trained DRIFT-RL model
```bash
python inc_env.py eval --model drift_final.zip --out rl_results.csv
```
### Run only baselines
```bash
python inc_env.py baselines --out baseline_results.csv
```
### Run both DRIFT-RL and baselines, merge results
```bash
python inc_env.py all --model drift_final.zip --out all_results.csv
```

