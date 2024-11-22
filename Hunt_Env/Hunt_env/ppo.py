import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 定义全连接层
        self.fc = nn.Linear(input_dim, 128)
        # 定义actor层
        self.actor = nn.Linear(128, action_dim)
        # 定义critic层
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        # 前向传播，使用ReLU激活函数
        x = torch.relu(self.fc(x))
        # 计算动作概率
        action_probs = torch.softmax(self.actor(x), dim=-1)
        # 计算状态值
        state_value = self.critic(x)
        return action_probs, state_value

class PPO:
    def __init__(self, env, agent_type, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.env = env
        self.agent_type = agent_type
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # 根据智能体类型初始化策略网络和优化器
        if agent_type == 'hunter':
            input_dim = env.observation_space['agent_0']['agent_obs']['position'].shape[0] + \
                        env.observation_space['agent_0']['agent_obs']['velocity'].shape[0] + \
                        env.observation_space['agent_0']['agent_obs']['accel'].shape[0] + \
                        env.observation_space['agent_0']['agent_obs']['angle'].shape[0] + \
                        env.observation_space['agent_0']['agent_obs']['angle_velocity'].shape[0]
            action_dim = env.action_space['agent_0'].nvec[0]
        else:
            input_dim = env.observation_space['agent_0']['agent_obs']['position'].shape[0] + \
                        env.observation_space['agent_0']['agent_obs']['velocity'].shape[0] + \
                        env.observation_space['agent_0']['agent_obs']['accel'].shape[0] + \
                        env.observation_space['agent_0']['agent_obs']['angle'].shape[0] + \
                        env.observation_space['agent_0']['agent_obs']['angle_velocity'].shape[0]
            action_dim = env.action_space['agent_0'].nvec[0]

        self.policy = ActorCritic(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(input_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # 选择动作
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        # 计算奖励的蒙特卡洛估计
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 奖励归一化
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 将列表转换为张量
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs).to(device), 1).detach()

        # 优化策略网络
        for _ in range(self.k_epochs):
            # 评估旧动作和状态值
            logprobs, state_values = self.policy.evaluate(old_states, old_actions)

            # 计算比率 (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算代理损失
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)

            # 执行梯度下降
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略网络的权重
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        # 清空记忆
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]