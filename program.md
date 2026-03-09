# Hashfront Autonomous Research Program

Your goal is to maximize the win rate of an AI agent playing the turn-based tactics game "Hashfront" against the heuristic `BalancedStrategy`. 

## Instructions & Constraints:

1. **Modify Only `train.py`**: Iterate on the neural network architecture, the RL algorithm (e.g., PPO, REINFORCE, DQN), the observation space, and the reward function.
2. **Remove the Dummy Logic**: The current `train.py` ignores the neural network outputs and uses a hardcoded "Perfect Turtle" strategy (standing still and firing). You MUST replace this. The agent's actions must be sampled from the policy network's logits.
3. **Mandatory GPU Utilization**: The current baseline runs entirely on the CPU. You MUST update `train.py` to use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Ensure all models, observations, and tensors are moved to the GPU to maximize simulation throughput and training speed.
4. **Time Budget**: Training must run for exactly 5 minutes (300 seconds) wall-clock time. Do not change this limit.
5. **Target Metric**: At the end of the 5 minutes, print `Final win rate: X.XXXX`. Your sole objective is to push this number as high as possible.

Begin iterating on `train.py`. Run the 5-minute training block and evaluate the new win rate. If the changes improve the win rate, keep them. If performance degrades, revert the changes.