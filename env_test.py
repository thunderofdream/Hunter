#测试代码
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.version)

# from gymnasium.spaces import MultiDiscrete
# import numpy as np
# action_space = MultiDiscrete([5, 8])
# print(action_space.sample())
import time
import gymnasium
import Hunt_env
from gymnasium.wrappers import FlattenObservation
from Hunt_env.envs.decision.rule_based_decision import rule_based_decision
from gymnasium.wrappers import RecordVideo, Autoreset

# gymnasium.pprint_registry()
# import gymnasium as gym
# # env = gym.make_vec("Hunt_env/HuntEnv-v0", num_envs=3, render_mode='rgb_array')
# env = gym.make("Hunt_env/HuntEnv-v0", render_mode='rgb_array')
# env = Autoreset(env)
# # env = RecordVideo(env, video_folder='videos', episode_trigger=lambda x: True)
# print(env)
# # 重置环境，获取初始观测值
# obs = env.reset()

# # 初始化 replay buffer
# replay_buffer = []

# epoch = 0
# while epoch < 10:
#     # 随机选择一个动作
#     actions = env.action_space.sample()
    
#     # 执行动作，获取新的观测值、奖励、终止标志和其他信息
#     next_obs, rewards, terminated, trunccated, infos = env.step(actions)
    
#     # 将数据填充到 replay buffer 中

        
    
#     # 更新观测值
#     obs = next_obs

    
#     if terminated or trunccated:
#         epoch += 1
#         obs = env.reset()
#         terminated, trunccated = False, False

# # 关闭环境
# env.close()

env = gymnasium.make("Hunt_env/HuntEnv-v0",render_mode='human')

terminated, trunccated, info = False, False, {}
obs,info = env.reset()
# while not terminated and not trunccated:
while True:
    action = env.action_space.sample()
    action_1 = {'agent_{}'.format(i): rule_based_decision(obs['agent_{}'.format(i)], agent.cn_name) 
              for i, agent in enumerate(env.env.env.agents)}
    
    obs, reward, terminated, trunccated, info = env.step(action)
    env.render()
    print(obs)
    print(reward)
    print(terminated)
    print(info)
    print(env.env.env.current_step)
    print('-----------------')
    if terminated or trunccated:
        obs,info = env.reset()

    # time.sleep(1)
# wrapped_env = FlattenObservation(env)
# print(wrapped_env.reset())