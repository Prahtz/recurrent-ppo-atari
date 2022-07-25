from collections import defaultdict

class AverageReward:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.episodic_returns = [0.0] * num_envs

    
    def __call__(self, rewards, dones):
        assert rewards.shape[0] == self.num_envs
        assert dones.shape == rewards.shape

        returns = defaultdict(list)
        time_steps = rewards.shape[1]
        for i in range(self.num_envs):
            for t in range(time_steps):
                if dones[i][t]:
                    returns[t].append(self.episodic_returns[i]) 
                    self.episodic_returns[i] = 0.0
                else:
                    self.episodic_returns[i] += rewards[i][t]
        

        for t in returns.keys():
            avg_return = 0.0
            for r in returns[t]:
                avg_return += r
            avg_return /= len(returns[t])
            returns[t] = avg_return

        return returns