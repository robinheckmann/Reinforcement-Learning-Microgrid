from gym.envs.registration import register

register(
    id="environmentSofia/GridWorld-v0",
    entry_point="environmentSofia.envs:EMS",
)
