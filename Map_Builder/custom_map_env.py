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

    while len(possible_spawn_spot) < 1:
        pos_x = randint(border_size, col_size-border_size)
        pos_y = randint(border_size, row_size-border_size)

        test_spot = env[pos_y - 3:pos_y + 4, pos_x - 3:pos_x + 4]  # We check a 7x7 pixel patch around the agent. MAYBE CORRECT??
        test_spot_array = asarray(test_spot)  # We convert the patch to a array
        if test_spot_array.sum() == 12495:
            possible_spawn_spot.append([pos_x, pos_y])

    return possible_spawn_spot




class WorldEnv(gym.Env):
    def __init__(self):

        # RESET PARAMETERS
        self.agent_step = 0
        self.maxstep = 300

        # MAP PARAMETERS
        self.GLOBAL_MAP_SIZE = 750
        self.NUM_OBS = 15
        self.BORDER_SIZE = 50
        self.global_map = np.ones((self.GLOBAL_MAP_SIZE, self.GLOBAL_MAP_SIZE, 1), np.uint8)
        self.slam_map = self.global_map.copy()

        # AGENT PARAMETERS
        self.agent_pos_x = 0
        self.agent_pos_y = 0
        self.agent_size = 5
        self.agent_color = 100
        self.agent_range = 25
        self.agent_step_size = 5

        # --- OBSERVATION AND ACTION SPACE ---
        # Definition of observation space. We input pixel values between 0 - 255
        self.low = np.array(
            [0], dtype=np.int32
        )
        self.high = np.array(
            [255], dtype=np.int32
        )

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.int32)

        # Definition of action space.
        self.action_space = spaces.Discrete(8)
        self.action_dic = {0: (-self.agent_step_size, 0),
                           1: (self.agent_step_size, 0),
                           2: (0, -self.agent_step_size),
                           3: (0, self.agent_step_size),
                           4: (self.agent_step_size, self.agent_step_size),
                           5: (-self.agent_step_size, -self.agent_step_size),
                           6: (self.agent_step_size, -self.agent_step_size),
                           7: (-self.agent_step_size, self.agent_step_size)}

    def reset(self):
        # We reset the step
        self.agent_step = 0

        # We collect the generated map
        self.global_map = init_map(self.GLOBAL_MAP_SIZE, self.NUM_OBS, self.BORDER_SIZE)

        # SLAM MAP creation
        self.slam_map = self.global_map.copy()
        self.slam_map.fill(150)

        # We get a collection of possible spawn-points
        possible_spawn_points = select_agent_pos(self.global_map, self.BORDER_SIZE)

        # We draw a random spawn-point and draw it on the map
        self.agent_pos_x, self.agent_pos_y = possible_spawn_points[0]
        pos_x = self.agent_pos_x
        pos_y = self.agent_pos_y

        # StartY:EndY, StartX:EndX. Initial visible area for the SLAM map
        crop_img = self.global_map[pos_y - self.agent_range:pos_y + self.agent_range,
                                   pos_x - self.agent_range:pos_x + self.agent_range]

        # We add the initial visible area to the slam map
        self.slam_map[pos_y - self.agent_range:pos_y + self.agent_range,
                      pos_x - self.agent_range:pos_x + self.agent_range] = crop_img

        # We add the agent to the global map
        cv2.circle(self.global_map, (self.agent_pos_x, self.agent_pos_y), self.agent_size, self.agent_color, -1)

        return self.slam_map

    def step(self, action):
        # --- Step related variables ---
        collision = False  # To check if we're done
        done = False
        reward = 0

        self.agent_step += 1
        if self.agent_step == self.maxstep:  # If the agent has taken a certain number of steps we reset
            done = True

        # For removal of the previous position on the global map
        pre_agent_x = self.agent_pos_x
        pre_agent_y = self.agent_pos_y
        cv2.circle(self.global_map, (pre_agent_x, pre_agent_y), self.agent_size, 255, -1)  # Remove previous global position
        cv2.circle(self.slam_map, (pre_agent_x, pre_agent_y), self.agent_size, 255, -1)  # Remove previous slam position
        old_slam_map = self.slam_map.copy()

        # --- Defining movement ---
        move_x, move_y = self.action_dic[action]
        self.agent_pos_x += move_x
        self.agent_pos_y += move_y

        # --- Updating position ---
        # Adding new area to SLAM map
        pos_x = self.agent_pos_x
        pos_y = self.agent_pos_y

        # Checking collision
        test_spot = self.global_map[pos_y - 3:pos_y + 4,
                                    pos_x - 3:pos_x + 4]  # We check a 7x7 pixel patch around the agent. MAYBE CORRECT??
        test_spot_array = asarray(test_spot)  # We convert the patch to a array
        if test_spot_array.sum() != 12495:
            collision = True
            done = True

        # New visible area for the SLAM map
        crop_img = self.global_map[pos_y - self.agent_range:pos_y + self.agent_range,
                                   pos_x - self.agent_range:pos_x + self.agent_range]

        # We add the new visible area to the slam map
        self.slam_map[pos_y - self.agent_range:pos_y + self.agent_range,
                      pos_x - self.agent_range:pos_x + self.agent_range] = crop_img

        # Checking difference
        diff = cv2.absdiff(old_slam_map, self.slam_map)
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        crop_img = thresh[pos_y - self.agent_range:pos_y + self.agent_range,
                          pos_x - self.agent_range:pos_x + self.agent_range]
        diff = asarray(crop_img)

        # We add the new position of the agent to the global and slam map
        cv2.circle(self.global_map, (self.agent_pos_x, self.agent_pos_y), self.agent_size, self.agent_color, -1)
        cv2.circle(self.slam_map, (self.agent_pos_x, self.agent_pos_y), self.agent_size, self.agent_color, -1)


        # Defining reward
        if collision:
            reward -= 100
        else:
            num = diff.sum()/63750
            if num <= 0:
                reward -= 1
            else:
                reward += round(num, 2)

        return self.slam_map, done, reward

    def render(self):
        cv2.imshow("Global Map", self.global_map)
        cv2.imshow("SLAM Map", self.slam_map)


    def close(self):
        cv2.destroyAllWindows()
        quit()


world = WorldEnv()
done = False
for i in range(100):
    done = False
    state = world.reset()
    world.render()
    while not done:
        num = randint(0, 7)
        _, done, reward = world.step(0)
        print(reward)
        world.render()
        cv2.waitKey(500)


"""
# SLAM map
slam_map = global_map.copy()
slam_map.fill(150)
cv2.imshow("SLAM Map", slam_map)
cv2.waitKey()


# New observation
        self.observation = self.slam_map[pos_y - self.agent_range - 10:pos_y + self.agent_range + 10,
                                         pos_x - self.agent_range - 10:pos_x + self.agent_range + 10]

        row_size, col_size, _ = self.observation.shape
        #cv2.circle(self.observation, (int(col_size / 2), int(row_size / 2)), self.agent_size, self.agent_color, -1)
"""
