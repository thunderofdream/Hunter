from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import yaml
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from Hunt_env.envs.agent.agent_hunter import AgentHunter, AgentEscaper
import math



# class Actions(Enum):
#     right = 0
#     up = 1
#     left = 2
#     down = 3


class HunterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],
                 "render_fps": 20,
                 "max_episode_steps": 100}

    def __init__(self, render_mode=None, size=5):

        # 读取配置文件
        with open(r'Hunt_Env\Hunt_env\envs\env_config\env_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        # 读取配置文件中的参数
        self.num_hunters = config['Env_config']['num_hunters']
        self.num_escapers = config['Env_config']['num_escapers']
        self.num_agents = self.num_hunters + self.num_escapers
        self.TIMESCALE = config['Env_config']['TIMESCALE']
        self.map_size = config['Env_config']['map_size']/2.0
        self.max_step = config['Env_config']['max_steps']
        HunterEnv.metadata["max_episode_steps"] = self.max_step
        self.current_step = 0
        self.window_size = 512
        self.frames = []  # 用于存储渲染帧



        # 初始化智能体
        self.hunters = []
        self.escapers = []
         # 读取配置文件并创建AgentHunter实例
        with open(r'Hunt_Env\Hunt_env\envs\agent\agent_hunter.yaml', 'r',encoding='utf-8') as file:
            config_hunter = yaml.safe_load(file)['hunter1']
            config_hunter['map_size'] = self.map_size
            self.kill_range = config_hunter['kill_range']
        with open(r'Hunt_Env\Hunt_env\envs\agent\agent_escaper.yaml', 'r',encoding='utf-8') as file:
            config_escaper = yaml.safe_load(file)['escaper1']
            config_escaper['map_size'] = self.map_size

        for _ in range((self.num_hunters)):
            self.hunters.append(AgentHunter(config_hunter, TIMESCALE=self.TIMESCALE))
        for _ in range(self.num_escapers):
            self.escapers.append(AgentEscaper(config_escaper, TIMESCALE=self.TIMESCALE))

        self.agents = self.hunters + self.escapers


        # 定义动作空间和观测空间
        self.action_space = Dict({
            'agent_{}'.format(i): MultiDiscrete([5, 9]) for i in range(self.num_agents)
        })
        self.observation_space = Dict({
            'agent_{}'.format(i): 
            Dict({
                'agent_obs': Dict({
                    'position': Box(low=-self.map_size, high=self.map_size, shape=(2,), dtype=np.float32),
                    'velocity': Box(low=agent.speed_min, high=agent.speed_max, shape=(1,), dtype=np.float32),
                    'accel': Box(low=-1*agent.accel_max, high=agent.accel_max, shape=(1,), dtype=np.float32),
                    'angle': Box(low=0., high=math.pi*2, shape=(1,), dtype=np.float32),
                    'angle_velocity': Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32)
                }),
                'other_agent_obs': Dict({
                    'agent_{}'.format(j): Dict({
                        'position': Box(low=-self.map_size, high=self.map_size, shape=(2,), dtype=np.float32),
                        'velocity': Box(low=agent.speed_min, high=agent.speed_max, shape=(1,), dtype=np.float32),
                        'angle': Box(low=0., high=math.pi*2, shape=(1,), dtype=np.float32)
                    }) for j in range(self.num_agents) if j != i
                })
            }) for i, agent in enumerate(self.agents)
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        # 获取智能体的观测
        obs = {}
        for i, agent in enumerate(self.agents):
            agent_obs = agent.get_observation()
            other_agents_obs = {}
            for j, other_agents in enumerate(self.agents):
                if i != j:
                    if agent.isFind_sector(other_agents, agent.explore_size):
                        other_agents_obs['agent_{}'.format(j)] = {
                            'position': np.array(other_agents.position, dtype=np.float32),
                            'velocity': np.array([other_agents.velocity], dtype=np.float32),
                            'angle': np.array([other_agents.angle], dtype=np.float32)
                        }
                    else:
                        other_agents_obs['agent_{}'.format(j)] = {
                            'position': np.zeros(2, dtype=np.float32),
                            'velocity': np.zeros(1, dtype=np.float32),
                            'angle': np.zeros(1, dtype=np.float32)
                        }
            obs['agent_{}'.format(i)] = {
                'agent_obs': agent_obs,
                'other_agent_obs': other_agents_obs
            }
        return obs

    def _get_info(self):
        # 获取智能体的信息
        info = {}
        for i, agent in enumerate(self.agents):
            info['agent_{}'.format(i)] = {
                '阵营': agent.cn_name,
                'position': agent.position,
                'velocity': agent.velocity,
                'angle': agent.angle
            }
        info['reward'] = self.get_reward()
        return info

    def reset(self, seed=None, options=None):
        # 重置环境
        for agent in self.agents:
            agent.reset()
        self.current_step = 0
        return self._get_obs(), self._get_info()

    def step(self, action):

        # 执行动作
        for i, agent in enumerate(self.agents):
            agent.action(action['agent_{}'.format(i)])
        # 更新智能体的状态
        for agent in self.agents:
            agent.update()

        # 判断是否终止
        self.current_step += 1
        terminated = self._is_terminal()

        # 判断是否���断
        trunccated = False
        if self.current_step >= self.max_step:
            trunccated = True

        # 获取奖励
        rewards_dict = self.get_reward()
        rewards = sum(rewards_dict.values())
        return self._get_obs(), rewards, terminated, trunccated, self._get_info()

    def get_reward(self):
        rewards = {}
        for i, agent in enumerate(self.agents):
            if agent.cn_name == '猎手':
                # 猎手的奖励逻辑
                rewards['agent_{}'.format(i)] = self._calculate_hunter_reward(agent)
            else:
                # 逃脱者的奖励逻辑
                rewards['agent_{}'.format(i)] = self._calculate_escaper_reward(agent)
        return rewards

    def _calculate_hunter_reward(self, agent):
        reward = 0
        for other_agent in self.agents:
            if other_agent.cn_name == '逃脱者':
                distance = agent.distance(other_agent)
                reward += agent.explore_size - distance  # 距离越近奖励越大
        return reward

    def _calculate_escaper_reward(self, agent):
        reward = 0
        for other_agent in self.agents:
            if other_agent.cn_name == '猎手':
                distance = agent.distance(other_agent)
                reward += distance - agent.explore_size  # 距离越远奖励越大
        return reward
    
    def _is_terminal(self):
        for escaper in self.escapers:
            for hunter in self.hunters:
                if hunter.distance(escaper) < hunter.kill_range:
                    escaper.is_killed = True
        return all(escaper.is_killed for escaper in self.escapers)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human" or self.render_mode == "rgb_array":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # 绘制地图
        map_size = 200
        map_scale = self.window_size / map_size
        pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(0, 0, self.window_size, self.window_size), 1)

        # 绘制智能体及其探测范围和轨迹
        for agent in self.agents:
            color = agent.color
            faded_color = agent.faded_color
            pos = (self.window_size / 2 + float(agent.position[0]) * map_scale, self.window_size / 2 - float(agent.position[1]) * map_scale)

            # 创建一个具有透明度的 Surface
            transparent_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            transparent_surface = transparent_surface.convert_alpha()

            # 确保 faded_color 是一个包含 RGB 值的 NumPy 数组，并添加透明度值
            faded_color_with_alpha = np.append(faded_color, 128)

            # 绘制具有透明度的圆
            pygame.draw.circle(transparent_surface, faded_color_with_alpha, pos, float(agent.explore_size * map_scale), 0)
            canvas.blit(transparent_surface, (0, 0))

            # 绘制智能体位置
            pygame.draw.circle(canvas, color, pos, 1.5 * map_scale)

            # 绘制轨迹
            if len(agent.trajectory) > 1:
                pygame.draw.lines(canvas, color, False, [(self.window_size / 2 + p[0] * map_scale, self.window_size / 2 - p[1] * map_scale) for p in agent.trajectory], 1)

        # 绘制图例
        legend_x = self.window_size - 100
        legend_y = 10
        pygame.draw.rect(canvas, (255, 255, 255), pygame.Rect(legend_x, legend_y, 90, 100))
        pygame.draw.circle(canvas, self.hunters[0].color, (legend_x + 10, legend_y + 10), 5)
        pygame.draw.circle(canvas, self.escapers[0].color, (legend_x + 10, legend_y + 30), 5)

        # 使用系统字体
        font = pygame.font.SysFont('SimHei', 15)  # 使用黑体字体

        # 绘制汉字
        text_surface = font.render("猎手", True, (0, 0, 0))
        canvas.blit(text_surface, (legend_x + 20, legend_y + 5))
        text_surface = font.render("逃脱者", True, (0, 0, 0))
        canvas.blit(text_surface, (legend_x + 20, legend_y + 25))

        # 绘制对局信息
        info_y = legend_y + 50
        for i, agent in enumerate(self.agents):
            text_surface = font.render(f"Agent {i}: Pos {agent.position}, Reward {self.get_reward()['agent_{}'.format(i)]}", True, (0, 0, 0))
            canvas.blit(text_surface, (10, info_y))
            info_y += 20

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            frame = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
            self.frames.append(frame)  # 保存帧
            return frame

    def save_video(self, filename):
        import imageio
        imageio.mimsave(filename, self.frames, fps=self.metadata["render_fps"])
        self.frames = []  # 清空帧缓存

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_agent_dims(self, agent_id):
        agent = self.agents[agent_id]
        input_dim = self.observation_space['agent_{}'.format(agent_id)]['agent_obs']['position'].shape[0] + \
                    self.observation_space['agent_{}'.format(agent_id)]['agent_obs']['velocity'].shape[0] + \
                    self.observation_space['agent_{}'.format(agent_id)]['agent_obs']['accel'].shape[0] + \
                    self.observation_space['agent_{}'.format(agent_id)]['agent_obs']['angle'].shape[0] + \
                    self.observation_space['agent_{}'.format(agent_id)]['agent_obs']['angle_velocity'].shape[0] + \
                    sum(self.observation_space['agent_{}'.format(agent_id)]['other_agent_obs']['agent_{}'.format(j)]['position'].shape[0] +
                        self.observation_space['agent_{}'.format(agent_id)]['other_agent_obs']['agent_{}'.format(j)]['velocity'].shape[0] +
                        self.observation_space['agent_{}'.format(agent_id)]['other_agent_obs']['agent_{}'.format(j)]['angle'].shape[0]
                        for j in range(self.num_agents) if j != agent_id)
        action_dim1 = self.action_space['agent_{}'.format(agent_id)].nvec[0]
        action_dim2 = self.action_space['agent_{}'.format(agent_id)].nvec[1]
        return input_dim, action_dim1, action_dim2

