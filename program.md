# Autoresearch Goal: Hashfront AI Optimization

Your objective is to modify `train.py` to train an autonomous agent that can beat the `BalancedStrategy` in the Hashfront game.

## Instructions
1. Review the existing `train.py` and the game simulator at `../hashfront/tools/simulator.py`.
2. Enhance the neural network architecture (e.g., add convolutional layers, self-attention, or recurrent networks to parse the grid state).
3. Improve the reinforcement learning algorithm. Implement a robust method like PPO (Proximal Policy Optimization) or REINFORCE.
4. Experiment with different hyperparameters (learning rate, discount factor, entropy regularization, etc.).
5. Ensure your modified `train.py` runs training and evaluation for exactly 5 minutes.
6. The script MUST output the final win rate at the very end of execution in a clear format (e.g., `Final win rate: X.XXXX`).

Maximize the final win rate against `BalancedStrategy`.
