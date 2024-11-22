import gymnasium as gym
import torch
from Hunt_env.envs.Hunt_world import HunterEnv
from Hunt_env.envs.decision.rule_based_decision import rule_based_decision
from ppo import PPOAgent, Memory
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import FlattenObservation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_observation(obs):
    # 将观测转换为张量
    processed_obs = []
    for key, value in obs.items():
        if isinstance(value, dict):
            processed_obs.append(process_observation(value))
        else:
            tensor_value = torch.tensor(value, dtype=torch.float32).to(device)
            if tensor_value.dim() == 0:  # 如果是零维张量，转换为一维张量
                tensor_value = tensor_value.unsqueeze(0)
            processed_obs.append(tensor_value.view(-1,))
    return torch.cat(processed_obs)

def main():
    # 创建环境和PPO算法实例
    env = HunterEnv(render_mode='rgb_array')
    env = RecordVideo(env, video_folder='videos', episode_trigger=lambda x: x % 50 == 0)
    ppo_hunter = PPOAgent(env, agent_id=0, buffer_size=2048)  # 假设第一个智能体是猎手
    ppo_escaper = PPOAgent(env, agent_id=1, buffer_size=2048)  # 假设第二个智能体是逃脱者

    # 定义训练参数
    max_episodes = 5000
    max_timesteps = 300
    log_interval = 10

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        state_hunter = process_observation(state['agent_0'])
        state_escaper = process_observation(state['agent_1'])
        episode_reward_hunter = 0
        episode_reward_escaper = 0

        for t in range(max_timesteps):
            # 选择动作
            (action_hunter1, action_hunter2), (log_prob_hunter1, log_prob_hunter2) = ppo_hunter.select_action(state_hunter)
            (action_escaper1, action_escaper2), (log_prob_escaper1, log_prob_escaper2) = ppo_escaper.select_action(state_escaper)
            
            #逃脱者使用基于规则的策略

            # action_escaper1 = rule_based_decision(state['agent_1'], '逃脱者')


            action = {'agent_0': [action_hunter1.cpu(), action_hunter2.cpu()], 'agent_1': [action_escaper1.cpu(), action_escaper2.cpu()]}
            # action = {'agent_0': [action_hunter1.cpu(), action_hunter2.cpu()], 'agent_1': action_escaper1}
            state, _, done, _, info = env.step(action)
            state_hunter = process_observation(state['agent_0'])
            state_escaper = process_observation(state['agent_1'])

            # 存储记忆
            ppo_hunter.store_transition(state_hunter, torch.tensor([action_hunter1, action_hunter2]), torch.tensor([log_prob_hunter1, log_prob_hunter2]), info['reward']['agent_0'], done)
            ppo_escaper.store_transition(state_escaper, torch.tensor([action_escaper1, action_escaper2]), torch.tensor([log_prob_escaper1, log_prob_escaper2]), info['reward']['agent_1'], done)

            episode_reward_hunter += info['reward']['agent_0']
            episode_reward_escaper += info['reward']['agent_1']

            if done:
                break

        # 更新策略
        if len(ppo_hunter.memory.states) >= ppo_hunter.buffer_size:
            ppo_hunter.update()
            ppo_hunter.clear_memory()

        if len(ppo_escaper.memory.states) >= ppo_escaper.buffer_size:
            ppo_escaper.update()
            ppo_escaper.clear_memory()

        # 打印日志
        if episode % log_interval == 0:
            print(f'第 {episode} 集 \t 猎手奖励: {episode_reward_hunter} \t 逃脱者奖励: {episode_reward_escaper}')

    env.close()
    ppo_hunter.save('hunter.pth')
    ppo_escaper.save('escaper.pth')

if __name__ == '__main__':
    main()