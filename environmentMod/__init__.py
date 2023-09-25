from gym.envs.registration import register

register(
    id="environmentMod/GridWorld-v0",
    entry_point="environmentMod.envs:EMS",
)
