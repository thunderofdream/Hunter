import numpy as np

def hunter_decision(obs):
    # 猎手的决策逻辑
    action = [3, 4]  # 初始化动作，加速度为0，角速度为0°
    hunter_obs = obs['agent_obs']
    other_agents_obs = obs['other_agent_obs']
    
    # 寻找最近的逃脱者
    min_distance = float('inf')
    target_position = None
    for agent_id, agent_obs in other_agents_obs.items():
        if agent_obs['position'].any():
            distance = np.linalg.norm(hunter_obs['position'] - agent_obs['position'])
            if distance < min_distance:
                min_distance = distance
                target_position = agent_obs['position']
    
    if target_position is not None:
        # 计算加速度和角速度
        direction = target_position - hunter_obs['position']
        angle_to_target = np.arctan2(direction[1], direction[0])
        angle_diff = angle_to_target - hunter_obs['angle'][0]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # 将角度差限制在[-π, π]范围内
        
        # 根据角度差选择角速度动作
        if angle_diff > 0:
            if angle_diff < np.pi / 8:
                action[1] = 4  # 0°
            elif angle_diff < 3 * np.pi / 8:
                action[1] = 5  # 22.5°
            elif angle_diff < 5 * np.pi / 8:
                action[1] = 6  # 45°
            elif angle_diff < 7 * np.pi / 8:
                action[1] = 7  # 67.5°
            else:
                action[1] = 8  # 90°
        else:
            if angle_diff > -np.pi / 8:
                action[1] = 4  # 0°
            elif angle_diff > -3 * np.pi / 8:
                action[1] = 3  # -22.5°
            elif angle_diff > -5 * np.pi / 8:
                action[1] = 2  # -45°
            elif angle_diff > -7 * np.pi / 8:
                action[1] = 1  # -67.5°
            else:
                action[1] = 0  # -90°
        
        # 根据距离选择加速度动作
        if min_distance > 10:
            action[0] = 4  # 最大加速度
        elif min_distance > 5:
            action[0] = 3
        elif min_distance > 2:
            action[0] = 2
        elif min_distance > 1:
            action[0] = 1
        else:
            action[0] = 0  # 最小加速度
    else:
        # 如果没有找到逃脱者，随机选择动作
        action[1] = np.random.randint(9)
    
    return np.array(action, dtype=np.int64)

def escaper_decision(obs):
    # 逃脱者的决策逻辑
    action = [3, 4]  # 初始化动作，加速度为0，角速度为0°
    escaper_obs = obs['agent_obs']
    other_agents_obs = obs['other_agent_obs']
    
    # 寻找最近的猎手
    min_distance = float('inf')
    threat_position = None
    for agent_id, agent_obs in other_agents_obs.items():
        if agent_obs['position'].any():
            distance = np.linalg.norm(escaper_obs['position'] - agent_obs['position'])
            if distance < min_distance:
                min_distance = distance
                threat_position = agent_obs['position']
    
    if threat_position is not None:
        # 计算加速度和角速度
        direction = escaper_obs['position'] - threat_position
        angle_to_threat = np.arctan2(direction[1], direction[0])
        angle_diff = angle_to_threat - escaper_obs['angle'][0]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # 将角度差限制在[-π, π]范围内
        
        # 根据角度差选择角速度动作
        if angle_diff > 0:
            if angle_diff < np.pi / 8:
                action[1] = 4  # 0°
            elif angle_diff < 3 * np.pi / 8:
                action[1] = 5  # 22.5°
            elif angle_diff < 5 * np.pi / 8:
                action[1] = 6  # 45°
            elif angle_diff < 7 * np.pi / 8:
                action[1] = 7  # 67.5°
            else:
                action[1] = 8  # 90°
        else:
            if angle_diff > -np.pi / 8:
                action[1] = 4  # 0°
            elif angle_diff > -3 * np.pi / 8:
                action[1] = 3  # -22.5°
            elif angle_diff > -5 * np.pi / 8:
                action[1] = 2  # -45°
            elif angle_diff > -7 * np.pi / 8:
                action[1] = 1  # -67.5°
            else:
                action[1] = 0  # -90°
        
        # 根据距离选择加速度动作
        if min_distance < 5:
            action[0] = 4  # 最大加速度
        elif min_distance < 10:
            action[0] = 3
        elif min_distance < 15:
            action[0] = 2
        elif min_distance < 20:
            action[0] = 1
        else:
            action[0] = 0  # 最小加速度
    
    return np.array(action, dtype=np.int64)

def rule_based_decision(obs, agent_type):
    if '猎手' in agent_type:
        return hunter_decision(obs)
    elif '逃脱者' in agent_type:
        return escaper_decision(obs)
    else:
        raise ValueError("Unknown agent type")