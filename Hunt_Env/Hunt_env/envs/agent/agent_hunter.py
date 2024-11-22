import numpy as np
import math
import random
import yaml
from copy import deepcopy


#工具函数
def norm(vec2d):
    # from numpy.linalg import norm
    # faster to use custom norm because we know the vectors are always 2D
    assert len(vec2d) == 2
    return math.sqrt(vec2d[0]*vec2d[0] + vec2d[1]*vec2d[1])



class AgentEnity:
    _id_counter = 0
    @classmethod
    def _get_next_id(cls):
        id = cls._id_counter
        cls._id_counter += 1
        return id
    def __init__(self, config, TIMESCALE=1):
        self.TIMESCALE = np.float32(TIMESCALE)
        self.accel_max = np.float32(config['accel_max'])  # 加速度最大值
        self.speed_max = np.float32(config['speed_max'])  # 速度最大值
        self.speed_min = np.float32(config['speed_min'])  # 速度最小值
        self.explore_size = np.float32(config['explore_size'])  # 探测范围
        self.map_size = np.float32(config['map_size'])  # 地图大小
        self.cn_name = config['cn_name']  # 中文名
        self.color = np.array(config['color'], dtype=np.float32)
        self.faded_color = np.array(config['faded_color'], dtype=np.float32)
        
        # 分别对应x,y的值
        self.position = np.array(config['position'], dtype=np.float32)  # 位置
        self.init_position = np.array(config['position'], dtype=np.float32)  # 初始位置
        self.velocity = np.float32(config['velocity'])  # 速度 
        self.init_velocity = np.float32(config['velocity'])  # 初始速度
        self.accel = np.float32(config['accel'])  # 加速度
        self.init_accel = np.float32(config['accel'])  # 初始加速度
        # 假定角速度固定，船只转向效能有限
        self.angle = np.float32(config['angle']*(math.pi/180))  # 角度
        self.init_angle = np.float32(config['angle']*(math.pi/180)) # 初始角度
        self.angle_velocity = np.float32(config['angle_velocity']*(math.pi/180))  # 角速度
        self.init_angle_velocity = np.float32(config['angle_velocity']*(math.pi/180)) # 初始角速度
        
        self.bound_angle_velocity = np.array([angle * (math.pi / 180) for angle in config['bound_angle_velocity']], dtype=np.float32)  # 角速度边界限制
        # 角度边界限制，防止一直向某个方向变化达到无穷大
        # 出现了移动不正常的问题，修改一下移动的逻辑，所有的角度都限制到0-2pi之间    
        self.bound_angle = np.array([0, 2 * math.pi], dtype=np.float32)  # 角度边界限制

        self.default_position = np.array(config['default_position'], dtype=np.float32)  # 默认位置
        self.trajectory = []  # 轨迹

        self.is_killed = False

        

        #以下代码暂时不用
        # #维护一个记录是否见过某个智能体的列表
        # self.obs_history = {}
        # #维护一个上次看见各个智能体的位置的字典
        # self.last_obs_position = {}
        # #维护一个上次看见各个智能体的角度的字典
        # self.last_obs_angle = {}
        # #维护一个上次看见各个智能体的速度的字典
        # self.last_obs_speed = {}
        # #维护一个上次看见各个智能体的距离的字典
        # self.last_distance = {}
        # #维护一个上次看见各个智能体的步数的字典
        # self.last_find_step = {}
    
    def get_observation(self):
        return {
            'position': np.array(self.position, dtype=np.float32),
            'velocity': np.array([self.velocity], dtype=np.float32),  # 将标量转换为一维numpy数组
            'accel': np.array([self.accel], dtype=np.float32),  # 将标量转换为一维numpy数组
            'angle': np.array([self.angle], dtype=np.float32),  # 将标量转换为一维numpy数组
            'angle_velocity': np.array([self.angle_velocity], dtype=np.float32)  # 将标量转换为一维numpy数组
        }

    #执行动作
    def action(self, action):
        # action: [accel, angle_velocity]
        self.accel = action[0] - 2
        self.angle_velocity = (action[1]-4) * math.pi/4 
        
    def update(self):
        if self.is_killed:
            return
        """ Update the position and velocity. """
        self.position = self.position.astype(float)
        #加速度约束
        self.accel_constraint()
        #角速度约束
        self.angle_velocity_constraint()

        #动作更新
        self.velocity += self.accel
        self.angle    += self.angle_velocity

        # 角度约束
        self.angle_constraint()
        # 速度约束
        self.velocity_constraint()

        self.position += self.moving()
        #位置约束
        self.position_constraint()

      
        self.trajectory.append(self.position.copy())

    # def get_simulate_position(self):
    #     draw_pos = [0,0]
    #     draw_pos[0] = window.width // 2
    #     draw_pos[1] = window.height // 2    
    #     draw_pos[0] = draw_pos[0] + self.position[0] *DRAW_SCALE
    #     draw_pos[1] = draw_pos[1] + self.position[1] *DRAW_SCALE
    #     return draw_pos

    def accel_constraint(self):
        if self.accel < -self.accel_max * self.TIMESCALE:
            self.accel = -self.accel_max * self.TIMESCALE
        if self.accel > self.accel_max * self.TIMESCALE:
            self.accel = self.accel_max * self.TIMESCALE

    
    def angle_velocity_constraint(self):
        if self.angle_velocity < self.bound_angle_velocity[0]:
            self.angle_velocity = self.bound_angle_velocity[0]
        if self.angle_velocity > self.bound_angle_velocity[1]:
            self.angle_velocity = self.bound_angle_velocity[1]

    def position_constraint(self):
        if self.position[0] < -self.map_size:
            self.position[0] = -self.map_size
        if self.position[0] > self.map_size:
            self.position[0] = self.map_size
        if self.position[1] < -self.map_size:
            self.position[1] = -self.map_size
        if self.position[1] > self.map_size:
            self.position[1] = self.map_size
    '''
    description: 获取两个方向速度变化的数值，便于后续计算
    param {*} self
    return {*}
    '''    
    def moving(self):
        delta_x = self.velocity*math.cos(self.angle)
        delta_y = self.velocity*math.sin(self.angle)
        return np.array([delta_x,delta_y])
    
    '''
    description: 设定角度约束，将角度范围控制在0-2pi之间
    param {*} self
    return {*}
    '''
    def angle_constraint(self):
        if self.angle < self.bound_angle[0] :
            self.angle += math.pi * 2
        if self.angle > self.bound_angle[1]:
            self.angle -= math.pi * 2

    '''
    description: 设定速度约束，如果超出了给定的速度范围，将其控制在速度范围间
    param {*} self
    return {*}
    '''
    def velocity_constraint(self):
        if self.velocity < self.speed_min * self.TIMESCALE:
            self.velocity = self.speed_min * self.TIMESCALE
        if self.velocity > self.speed_max * self.TIMESCALE:
            self.velocity = self.speed_max * self.TIMESCALE

    def distance(self, other):
        """ Computes the euclidean distance to another entity. """
        return norm(self.position - other.position)
    
   
    def isFind_sector(self, other, explore_size,sector_angle=math.pi/2):
        #explore_size :扇形搜索区域的半径
        #other:另外一个智能体
        #sector_angle :扇形搜索区域的角度
        #判断这个智能通是否能在本智能体的扇形搜索区域里探测到另外一个智能体
        # 1.计算两个智能体之间的距离
        distance = self.distance(other)
        # 2.计算两个智能体之��的角度
        angle = self.angle_to_target(other)
        # # 3.判断是否在扇形区域内
        # if distance < explore_size and angle < sector_angle/2 and angle > -1*sector_angle/2:
        #     return True
        # 将观测范围设置为圆形
        if distance < explore_size:
            return True
        return False
    
    def angle_to_target(self, other):
        """ Computes the angle to another entity. """
        to_angle = math.atan2(other.position[1] - self.position[1], other.position[0] - self.position[0]) - self.angle
        #将角度转到0-2pi之间
        if to_angle < 0:
            to_angle += math.pi * 2
        if to_angle > math.pi * 2:
            to_angle -= math.pi * 2
        return to_angle
        
    def reset(self):
        self.position = deepcopy(self.init_position)
        self.velocity = deepcopy(self.init_velocity)
        self.accel = deepcopy(self.init_accel)
        self.angle = deepcopy(self.init_angle)
        self.angle_velocity = deepcopy(self.init_angle_velocity)
        self.trajectory = []


class AgentHunter(AgentEnity):
    def __init__(self, config, TIMESCALE=1):
        super(AgentHunter, self).__init__(config, TIMESCALE)
        self.hunter_id = self._get_next_id()
        self.kill_range = config['kill_range']  # 击杀范围
        

class AgentEscaper(AgentEnity):
    def __init__(self, config, TIMESCALE=1):
        super(AgentEscaper, self).__init__(config, TIMESCALE)
        self.escaper_id = self._get_next_id()
        self.is_killed = False

    def reset(self):
        super().reset()
        self.is_killed = False