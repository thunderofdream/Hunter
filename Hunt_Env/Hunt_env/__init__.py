from gymnasium.envs.registration import register

register(
    id="Hunt_env/HuntEnv-v0",
    entry_point="Hunt_env.envs:HunterEnv",
)
