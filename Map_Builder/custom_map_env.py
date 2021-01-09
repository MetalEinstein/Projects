import math
import gym
from gym import spaces, logger
from random import seed
from random import randint
from PIL import Image
from gym.utils import seeding
import numpy as np
from numpy import asarray
import cv2


def init_map(size, num_obs, border_size):
    # Drawing a empty map
    global_map = np.ones((size, size, 1), np.uint8)  # Defining size of map
    global_map.fill(0)  # Drawing it white
    cv2.rectangle(global_map, (border_size, border_size), (size - border_size, size - border_size), 255,
                  -1)  # Setting world boundary

    # Filling the map with obstacles
    num_obstacles = randint(0, num_obs)
    for obstacles in range(num_obstacles + 1):
        obs_selected = randint(0, 1)  # We randomly select between two obstacle types
        obstacle = generate_obs(obs_selected)  # We get a random obstacle position and type

        if obs_selected == 0:
            cv2.rectangle(global_map, obstacle[0], obstacle[1], 0, -1)
        else:
            cv2.circle(global_map, (obstacle[0], obstacle[1]), obstacle[2], 0, -1)

    return global_map


def generate_obs(selection):
    obs_pos_x = randint(0, 600)
    obs_pox_y = randint(0, 600)
    obstacle_list = {0: ((obs_pos_x - 30, obs_pox_y - 30), (obs_pos_x + 30, obs_pox_y + 30)),
                     1: (obs_pos_x, obs_pox_y, 20)}
    obstacle = obstacle_list[selection]

    return obstacle


def select_agent_pos(env, border_size):

    row_size, col_size, _ = env.shape
    possible_spawn_spot = []

    while len(possible_spawn_spot) < 5:
        pos_x = randint(border_size, col_size-border_size)
        pos_y = randint(border_size, row_size-border_size)

        test_spot = env[pos_y - 3:pos_y + 4, pos_x - 3:pos_x + 4]  # We check a 7x7 pixel patch around the agent. MAYBE CORRECT??
        test_spot_array = asarray(test_spot)  # We convert the patch to a array
        if test_spot_array.sum() == 12495:
            possible_spawn_spot.append([pos_x, pos_y])

    return possible_spawn_spot




class WorldEnv(gym.Env):
    def __init__(self):

        # MAP PARAMETERS
        self.GLOBAL_MAP_SIZE = 600
        self.NUM_OBS = 15
        self.BORDER_SIZE = 30

        # AGENT PARAMETERS
        self.agent_pos_x = 0
        self.agent_pos_y = 0
        self.agent_size = 5
        self.agent_color = 100
        self.agent_range = 25

        # --- OBSERVATION AND ACTION SPACE ---
        # Definition of observation space. We input pixel values between 0 - 255
        self.low = np.array(
            [0], dtype=np.int32
        )
        self.high = np.array(
            [255], dtype=np.int32
        )

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.int32)

        # Definition of action space. Four discrete actions, up, down, left, right
        self.action_space = spaces.Discrete(4)

    def reset(self):
        # We collect the generated map
        global_map = init_map(self.GLOBAL_MAP_SIZE, self.NUM_OBS, self.BORDER_SIZE)

        # We get a collection of possible spawn-points
        possible_spawn_points = select_agent_pos(global_map, self.BORDER_SIZE)

        # We draw a random spawn-point among the selection and draw it on the map
        self.agent_pos_x, self.agent_pos_y = possible_spawn_points[randint(0, len(possible_spawn_points)-1)]
        cv2.circle(global_map, (self.agent_pos_x, self.agent_pos_y), self.agent_size, self.agent_color, -1)

        return global_map, self.agent_pos_x, self.agent_pos_y


    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass
        # quit()



world = WorldEnv()
for i in range(100):
    global_map, pos_x, pos_y = world.reset()

    # StartY:EndY, StartX:EndX
    crop_img = global_map[pos_y-world.agent_range:pos_y+world.agent_range,
                          pos_x-world.agent_range:pos_x+world.agent_range]

    cv2.imshow("chop", crop_img)
    cv2.imshow("Global Map", global_map)
    cv2.waitKey()

"""
# SLAM map
slam_map = global_map.copy()
slam_map.fill(150)
cv2.imshow("SLAM Map", slam_map)
cv2.waitKey()
"""