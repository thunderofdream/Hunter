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

    def evaluate(self, state, action):
        x = torch.relu(self.fc(state))
        action_probs1 = torch.softmax(self.actor1(x), dim=-1)
        action_probs2 = torch.softmax(self.actor2(x), dim=-1)
        dist1 = Categorical(action_probs1)
        dist2 = Categorical(action_probs2)
        action_logprobs1 = dist1.log_prob(action[:, 0])
        action_logprobs2 = dist2.log_prob(action[:, 1])
        state_value = self.critic(x)
        return action_logprobs1, action_logprobs2, state_value

class PPOAgent:
    def __init__(self, env, agent_id, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, buffer_size=2048):
        self.env = env.unwrapped
        self.agent_id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.buffer_size = buffer_size

        # 从环境中获取输入输出维度
        input_dim, action_dim1, action_dim2 = self.env.get_agent_dims(agent_id)

        self.policy = ActorCritic(input_dim, action_dim1, action_dim2).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(input_dim, action_dim1, action_dim2).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

        self.memory = Memory()

    def select_action(self, state):
        # 选择动作
        with torch.no_grad():
            action_probs1, action_probs2, _ = self.policy_old(state)
        dist1 = Categorical(action_probs1)
        dist2 = Categorical(action_probs2)
        action1 = dist1.sample()
        action2 = dist2.sample()
        return (action1, action2), (dist1.log_prob(action1), dist2.log_prob(action2))

    def update(self):
        # 计算奖励的蒙特卡洛估计
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 奖励归一化
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 将列表转换为张量
        old_states = torch.squeeze(torch.stack(self.memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs).to(device), 1).detach()

        # 优化策略网络
        for _ in range(self.k_epochs):
            # 评估旧动作和状态值
            logprobs1, logprobs2, state_values = self.policy.evaluate(old_states, old_actions)

            # 计算比率 (pi_theta / pi_theta__old)
            ratios1 = torch.exp(logprobs1 - old_logprobs[:, 0].detach())
            ratios2 = torch.exp(logprobs2 - old_logprobs[:, 1].detach())

            # 计算代理损失
            advantages = rewards - state_values.detach().squeeze()
            surr1_1 = ratios1 * advantages
            surr1_2 = ratios2 * advantages
            surr2_1 = torch.clamp(ratios1, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            surr2_2 = torch.clamp(ratios2, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            surr1 = torch.min(surr1_1, surr2_1)
            surr2 = torch.min(surr1_2, surr2_2)
            loss = -surr1.mean() - surr2.mean() + 0.5 * self.MseLoss(state_values.squeeze(), rewards)

            # 执行梯度下降
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新旧策略网络的权重
        self.policy_old.load_state_dict(self.policy.state_dict())

    def store_transition(self, state, action, log_prob, reward, is_terminal):
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(log_prob)
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(is_terminal)

    def clear_memory(self):
        self.memory.clear_memory()

    #保存模型
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

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