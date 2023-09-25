from gym.envs.registration import register

register(
    id="environmentAmmonia/GridWorld-v0",
    entry_point="environmentAmmonia.envs:EMS",
)
