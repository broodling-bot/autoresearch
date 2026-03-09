import sys
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hashfront", "tools")))
from simulator import GameState, load_map, list_maps, run_game, STRATEGIES, BalancedStrategy, MAX_ROUNDS, do_move, do_attack, do_capture, get_attack_targets, do_wait, can_capture

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

def get_observation(state: GameState, player: int):
    # Dummy observation
    return [state.round_num, len(state.player_units(player)), len(state.enemy_units(player))]

class RLStrategy:
    def __init__(self, policy_net):
        self.policy_net = policy_net
        self.saved_log_probs = []
        self.rewards = []
        
    def play_turn(self, state, player, rng):
        # A minimal dummy implementation just for syntax correctness
        units = state.player_units(player)
        for unit in units:
            if not unit.alive or unit.has_acted: continue
            
            obs = get_observation(state, player)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits = self.policy_net(obs_tensor)
            
            # Just do random valid moves to avoid crashing
            targets = get_attack_targets(state, unit)
            if targets:
                t = rng.choice(targets)
                do_attack(state, rng, unit, t)
            else:
                do_wait(unit)

def train(duration=300):
    start_time = time.time()
    obs_dim = 3
    act_dim = 4 # N/S/E/W or similar
    
    policy = PolicyNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    maps = list_maps()
    if not maps:
        print("No maps found.")
        return
        
    games_played = 0
    wins = 0
    
    # Train loop for specified duration
    while time.time() - start_time < duration:
        map_name = random.choice(maps)
        rl_strat = RLStrategy(policy)
        p2_strat = BalancedStrategy()
        
        # We need a proper hook into the simulator, but the simulator's run_game just expects objects with play_turn.
        # run_game(p1_strat, p2_strat, map_name, seed, verbose, replay)
        seed = random.randint(0, 1000000)
        
        try:
            result = run_game(rl_strat, p2_strat, map_name, seed, verbose=False, replay=False)
            games_played += 1
            if result.winner == 1:
                wins += 1
                rl_strat.rewards.append(1.0)
            else:
                rl_strat.rewards.append(-1.0)
                
            # Dummy optimization step
            optimizer.zero_grad()
            # If we had log probs...
            optimizer.step()
        except Exception as e:
            print(f"Error running game: {e}")
            break
            
        # Quick exit for testing, but we need it to run for duration
        if games_played % 10 == 0:
            print(f"Played {games_played} games, wins: {wins}, win rate: {wins/games_played:.2%}")
            
    print(f"Final win rate: {wins/max(1, games_played):.4f}")

if __name__ == '__main__':
    # Run for 5 minutes
    train(duration=300)
