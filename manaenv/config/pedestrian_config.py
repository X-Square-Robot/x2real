"""Default runtime config for pedestrian random-walk extension."""

JOINT_ROLE_KEYWORDS: dict[str, list[str]] = {
    "l_hip": [
        "leftupleg",
        "lthigh",
        "leftthigh",
        "lupleg",
        "lefthip",
        "lhip",
        "leftupperleg",
        "lupperleg",
        "lfemur",
        "leftfemur",
    ],
    "r_hip": [
        "rightupleg",
        "rthigh",
        "rightthigh",
        "rupleg",
        "righthip",
        "rhip",
        "rightupperleg",
        "rupperleg",
        "rfemur",
        "rightfemur",
    ],
    "l_knee": [
        "leftleg",
        "lcalf",
        "leftshin",
        "leftcalf",
        "lshin",
        "lknee",
        "leftknee",
        "lleg",
        "leftlowerleg",
        "llowerleg",
    ],
    "r_knee": [
        "rightleg",
        "rcalf",
        "rightshin",
        "rightcalf",
        "rshin",
        "rknee",
        "rightknee",
        "rleg",
        "rightlowerleg",
        "rlowerleg",
    ],
    "l_arm": ["leftarm", "lupperarm", "leftupperarm", "larm", "lhumerus", "lefthumerus"],
    "r_arm": ["rightarm", "rupperarm", "rightupperarm", "rarm", "rhumerus", "righthumerus"],
}


class PedestrianRuntimeConfig:
    """Centralized defaults for pedestrian simulation."""

    PARENT_PRIM = "/World/envs/env_0/Characters"

    DEFAULT_COUNT = 3
    DEFAULT_WALK_SPEED = 1.0
    DEFAULT_ARRIVAL_THRESHOLD = 0.5
    DEFAULT_WAYPOINT_TIMEOUT = 30.0
    DEFAULT_IDLE_DURATION_MIN = 1.0
    DEFAULT_IDLE_DURATION_MAX = 4.0
    DEFAULT_CLEARANCE_M = 0.3
    DEFAULT_PLAN_RETRY_COUNT = 8
    DEFAULT_ASTAR_MAX_ITERATIONS = 100_000
    DEFAULT_WALK_PHASE_RATE = 8.0
    DEFAULT_HIP_SWING_AMPLITUDE = 0.35
    DEFAULT_KNEE_SWING_AMPLITUDE = 0.50
    DEFAULT_ARM_SWING_AMPLITUDE = 0.25
    DEFAULT_ARM_REST_DOWN_ANGLE = 1.2

    PATH_SIMPLIFY_TOLERANCE_FACTOR = 3
    WAYPOINT_ADVANCE_THRESHOLD = 0.25

    REQUIRED_EXTENSIONS = [
        "isaacsim.replicator.agent.core",
    ]


PedestrianRuntimeConfig = PedestrianRuntimeConfig()
