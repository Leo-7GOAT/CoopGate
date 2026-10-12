from dataclasses import asdict, dataclass


@dataclass
class Config:
    n_agents: int = 3
    sim_dt: float = 0.1
    decision_dt: float = 0.5
    episode_seconds: float = 30.0
    road_length: float = 600.0
    merge_end: float = 220.0
    goal_x: float = 410.0
    lane_width: float = 3.2
    vehicle_length: float = 4.8
    vehicle_width: float = 1.8
    desired_speed: float = 18.0
    max_speed: float = 23.0
    max_accel: float = 2.5
    max_brake: float = 5.0
    max_jerk: float = 5.0
    lane_change_seconds: float = 2.0
    mpc_horizon: int = 12
    gate_horizon: int = 10
    safety_gap: float = 2.0
    gate_threshold: float = 0.8
    gate_return_threshold: float = 0.9
    switch_penalty: float = 0.08
    background_brake: float = 2.5
    prediction_error: float = 0.35

    def dict(self):
        return asdict(self)


ACTION_SPEEDS = (8.0, 14.0, 20.0)
N_ACTIONS = 9
OBS_DIM = 25
