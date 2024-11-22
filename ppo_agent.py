
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim1, action_dim2):
        super(ActorCritic, self).__init__()
        # 定义全连接层
        self.fc = nn.Linear(input_dim, 128)
        # 定义actor层
        self.actor1 = nn.Linear(128, action_dim1)
        self.actor2 = nn.Linear(128, action_dim2)
        # 定义critic层
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        # 前向传播，使用ReLU激活函数
        x = torch.relu(self.fc(x))
        # 计算动作概率
        action_probs1 = torch.softmax(self.actor1(x), dim=-1)
        action_probs2 = torch.softmax(self.actor2(x), dim=-1)
        # 计算状态值
        state_value = self.critic(x)
        return action_probs1, action_probs2, state_value

class PPOAgent:
    def __init__(self, env, agent_type, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.env = env
        self.agent_type = agent_type
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # 计算输入维度
        input_dim = self.calculate_input_dim(env)
        action_dim1 = env.action_space.spaces['agent_0'].nvec[0]
        action_dim2 = env.action_space.spaces['agent_0'].nvec[1]

        self.policy = ActorCritic(input_dim, action_dim1, action_dim2).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(input_dim, action_dim1, action_dim2).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def calculate_input_dim(self, env):
        agent_obs_space = env.observation_space.spaces['agent_obs']
        other_agent_obs_space = env.observation_space.spaces['other_agent_obs']
        input_dim = sum(space.shape[0] for space in agent_obs_space.spaces.values())
        input_dim += sum(space.shape[0] for space in other_agent_obs_space.spaces.values())
        return input_dim

    def select_action(self, state):
        # 选择动作
        with torch.no_grad():
            action_probs1, action_probs2, _ = self.policy_old(state)
        dist1 = Categorical(action_probs1)
        dist2 = Categorical(action_probs2)
        action1 = dist1.sample()
        action2 = dist2.sample()
        return (action1.item(), action2.item()), (dist1.log_prob(action1), dist2.log_prob(action2))

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
            logprobs1, logprobs2, state_values = self.policy.evaluate(old_states, old_actions)

            # 计算比率 (pi_theta / pi_theta__old)
            ratios1 = torch.exp(logprobs1 - old_logprobs[:, 0].detach())
            ratios2 = torch.exp(logprobs2 - old_logprobs[:, 1].detach())

            # 计算代理损失
            advantages = rewards - state_values.detach()
            surr1 = ratios1 * advantages
            surr2 = ratios2 * advantages
            surr = torch.min(surr1, surr2)
            loss = -surr + 0.5 * self.MseLoss(state_values, rewards)

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