import gymnasium as gym
import torch
from Hunt_env.envs.Hunt_world import HunterEnv
from ppo import PPO, Memory
from gymnasium.wrappers import RecordVideo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_observation(obs):
    # 将观测转换为张量
    processed_obs = []
    for key, value in obs.items():
        if isinstance(value, dict):
            processed_obs.extend(process_observation(value))
        else:
            processed_obs.append(torch.FloatTensor(value).to(device))
    return torch.cat(processed_obs)

def main():
    # 创建环境和PPO算法实例
    env = HunterEnv(render_mode='rgb_array')
    env = RecordVideo(env, video_folder='videos', episode_trigger=lambda x: x % 100 == 0)
    memory_hunter = Memory()
    memory_escaper = Memory()
    ppo_hunter = PPO(env, agent_type='hunter')
    ppo_escaper = PPO(env, agent_type='escaper')

    # 定义训练参数
    max_episodes = 1000
    max_timesteps = 300
    update_timestep = 2000
    log_interval = 10

    timestep = 0
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        state_hunter = process_observation(state['agent_0']['agent_obs'])
        state_escaper = process_observation(state['agent_1']['agent_obs'])
        episode_reward_hunter = 0
        episode_reward_escaper = 0

        for t in range(max_timesteps):
            timestep += 1

            # 选择动作
            action_hunter, log_prob_hunter = ppo_hunter.select_action(state_hunter)
            action_escaper, log_prob_escaper = ppo_escaper.select_action(state_escaper)
            action = {'agent_0': action_hunter, 'agent_1': action_escaper}
            state, reward, done, _, _ = env.step(action)
            state_hunter = process_observation(state['agent_0']['agent_obs'])
            state_escaper = process_observation(state['agent_1']['agent_obs'])

            # 存储记忆
            memory_hunter.states.append(state_hunter)
            memory_hunter.actions.append(action_hunter)
            memory_hunter.logprobs.append(log_prob_hunter)
            memory_hunter.rewards.append(reward['agent_0'])
            memory_hunter.is_terminals.append(done)

            memory_escaper.states.append(state_escaper)
            memory_escaper.actions.append(action_escaper)
            memory_escaper.logprobs.append(log_prob_escaper)
            memory_escaper.rewards.append(reward['agent_1'])
            memory_escaper.is_terminals.append(done)

            episode_reward_hunter += reward['agent_0']
            episode_reward_escaper += reward['agent_1']

            # 更新策略
            if timestep % update_timestep == 0:
                ppo_hunter.update(memory_hunter)
                memory_hunter.clear_memory()
                ppo_escaper.update(memory_escaper)
                memory_escaper.clear_memory()
                timestep = 0

            if done:
                break

        # 打印日志
        if episode % log_interval == 0:
            print(f'第 {episode} 集 \t 猎手奖励: {episode_reward_hunter} \t 逃脱者奖励: {episode_reward_escaper}')

    env.close()

if __name__ == '__main__':
    main()