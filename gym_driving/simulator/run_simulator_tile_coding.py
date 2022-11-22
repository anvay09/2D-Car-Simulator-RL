from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import time
from matplotlib.pyplot import angle_spectrum
from numpy import angle
import pygame, sys
from pygame.locals import *
import random
import math
import argparse
import os.path

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10000

# weight_file = 'weights_T2_H.npy' #performing quite well - parameters - 
# ranges = [[0,1000], [0, 10], [0, 360], [0, 1000], [0, 1000], [0, 1000], [-350, 350], [-350, 350]] # distance to road, velocity, angle, distance to obstacle {l, c, r}
# num_tilings = 4
# epsilon = 0.1
# bin_counts = [10, 3, 20, 3, 3, 3, 2, 4]

weight_file = 'weights_T2_L.npy'

def get_normalized_direction(angle):
    dx = math.cos(angle * math.pi / 180)
    dy = math.sin(angle * math.pi / 180)
    return dx, dy

def get_point(x, y, dx, dy, t):
    px = x + t*dx
    py = y + t*dy
    return px, py

def cast_ray(x, y, angle, theta, mud_pit_centres = []):
    dx, dy = get_normalized_direction(angle)

    ndx = dx * math.cos(theta * math.pi / 180) - dy * math.sin(theta * math.pi / 180)
    ndy = dx * math.sin(theta * math.pi / 180) + dy * math.cos(theta * math.pi / 180)
    
    for t in range(100):
        px, py = get_point(x, y, ndx, ndy, t)

        for centre in mud_pit_centres:
            x1 = centre[0]-50
            x2 = centre[0]+50
            y1 = centre[1]-50
            y2 = centre[1]+50

            if px >= x1 and px <= x2 and py >= y1 and py <= y2:
                # print('MUD PIT DETECTED, t = ', t)
                return t

        if px > 350 and py < 50 and py > -50 and px < 355: # road
            return 500

        elif px > 350 or px < -350 or py > 350 or py < -350:
            # print('WALL DETECTED, t = ', t)
            return t
    
    return 1000

def action_to_num(action):
    return action[0] * 5 + action[1]
    
def num_to_action(num):
    action = np.array([0,0]) 
    action[1] = num % 5
    action[0] = (num - action[1]) / 5
    return action

def create_tiling(feature_range, bins, offset):
    return np.linspace(feature_range[0], feature_range[1], bins + 1)[1:-1] + offset

def create_tilings(feature_ranges, number_tilings, bins, offsets):
    tilings = []
    # for each tiling
    for tile_i in range(number_tilings):
        tiling_bin = bins[tile_i]
        tiling_offset = offsets[tile_i]
        
        tiling = []
        # for each feature dimension
        for feat_i in range(len(feature_ranges)):
            feat_range = feature_ranges[feat_i]
            # tiling for 1 feature
            feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
            tiling.append(feat_tiling)
        tilings.append(tiling)
    return np.array(tilings)

def get_tile_coding(feature, tilings):
    num_dims = len(feature)
    feat_codings = []
    for tiling in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat_i = feature[i]
            tiling_i = tiling[i]  # tiling on that dimension
            coding_i = np.digitize(feat_i, tiling_i)
            feat_coding.append(coding_i)
        feat_codings.append(feat_coding)
    return np.array(feat_codings)

def get_features(state, mud_pit_centres = []):
    x = state[0]
    y = state[1]
    vel = state[2]
    angle = state[3]

    distance_to_road = np.sqrt((350-x)**2 + (y)**2)
    
    distance_to_obstacle_45 = cast_ray(x, y, angle, 45, mud_pit_centres)
    distance_to_obstacle_0 = cast_ray(x, y, angle, 0, mud_pit_centres)
    distance_to_obstacle_m45 = cast_ray(x, y, angle, -45, mud_pit_centres)
    # distance_to_obstacle_120 = cast_ray(x, y, angle, 120, mud_pit_centres)
    # distance_to_obstacle_m120 = cast_ray(x, y, angle, -120, mud_pit_centres)

    features = [distance_to_road, vel, angle, distance_to_obstacle_0, distance_to_obstacle_45, distance_to_obstacle_m45]
    return features

class QValueFunction:

    def __init__(self, tilings, actions, lr, weight_file = None):
        self.tilings = tilings
        self.num_tilings = len(self.tilings)
        self.actions = actions
        self.lr = lr  # /self.num_tilings  # learning rate equally assigned to each tiling
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in
                            self.tilings]  # [(10, 10), (10, 10), (10, 10)]
        if os.path.isfile(weight_file):
            print('Loading saved weights...')
            with open(weight_file, 'rb') as f:
                self.q_tables = np.load(f)
        else:
            self.q_tables = np.array([np.random.rand(*(state_size + (len(self.actions),))) for state_size in self.state_sizes])

    def value(self, state, action):
        state_codings = get_tile_coding(state, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = self.actions.index(action)

        value = 0
        for coding, q_table in zip(state_codings, self.q_tables):
            # for each q table
            value += q_table[tuple(coding) + (action_idx,)]
        return value / self.num_tilings

    def update(self, state, action, target):
        state_codings = get_tile_coding(state, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = self.actions.index(action)

        for coding, q_table in zip(state_codings, self.q_tables):
            delta = target - q_table[tuple(coding) + (action_idx,)]
            q_table[tuple(coding) + (action_idx,)] += self.lr * (delta)
        
    def change_lr(self, lr):
        self.lr = lr

    def save_weights(self, weight_file):
        with open(weight_file, 'wb') as f:
            np.save(f, self.q_tables)

    def display_values_for_action(self, state):
        for a in self.actions:
            val = self.value(state, a)
            print(state, a, val)

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, state, Q_tables, epsilon):
        """
        Input: The current state
        Output: Action to be taken
        """

        # Replace with your implementation to determine actions to be taken
        # x = state[0]
        # y = state[1]
        # vel = state[2]
        # angle = state[3]
        
        # if y > 0:
        #     required_angle = 360 - math.atan(y/(350-x)) * 180 / math.pi
        # else:
        #     required_angle = math.atan(-y/(350-x)) * 180 / math.pi

        # delta = angle - required_angle
        
        # if abs(delta) > 20:
        #     action_steer = 2
        #     action_acc = 2

        # else:
        #     action_steer = 1
        #     action_acc = 4

        if np.random.random() > epsilon:
            Q_vals = [Q_tables.value(state, a) for a in range(15)]
            translated_action = np.argmax(Q_vals)
            action = num_to_action(translated_action)
        else:
            action = num_to_action(np.random.randint(0, 15))


        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        global weight_file
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
    
        successes = 0
        ranges = [[0,1000], [0, 10], [0, 360], [0, 1000], [0, 1000], [0, 1000]] # distance to road, velocity, angle, distance to obstacle {l, c, r}
        num_tilings = 90
        epsilon = 0
        bin_counts = [10, 3, 20, 3, 3, 3]
        bins = [bin_counts for _ in range(num_tilings)]
        offsets = [[i/num_tilings * bin_counts[j] for j in range(len(bin_counts))] for i in range(num_tilings)]

        tilings = create_tilings(ranges, num_tilings, bins, offsets)
        actions = [a for a in range(15)]
        Q_tables = QValueFunction(tilings, actions, 0.1, weight_file)

        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
            print('Episode Num:', e)
            epsilon = epsilon / ((e/10)+1)

            for t in range(TIMESTEPS):
                
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                features = get_features(state)
                action = self.next_action(features, Q_tables, epsilon)
                translated_action = action_to_num(action)
                
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                
                new_features = get_features(state)
                
                if state[1] > 0:
                    required_angle = 2 * math.pi - math.atan(state[1]/(350-state[0]))
                else:
                    required_angle = math.atan(-state[1]/(350-state[0]))

                if new_features[0] < features[0] and new_features[1] > 1:
                    reward += 10 * math.cos(required_angle)
                else:
                    reward -= new_features[0] - features[0]
               
                # reward += math.cos(required_angle)

                if new_features[3] < 100:
                    reward -= (100 - new_features[3])/20
                    # print('OBSTACLE DETECTED LEFT')
                if new_features[4] < 100:
                    reward -= (100 - new_features[4])/5
                    # print('OBSTACLE DETECTED AHEAD')
                elif new_features[4] == 500 and new_features[1] > 1:
                    # print('POINTING TOWARDS ROAD!')
                    reward += 10
                if new_features[5] < 100:
                    reward -= (100 - new_features[5])/20
                    # print('OBSTACLE DETECTED RIGHT')
            
                # print(reward)

                target = reward + 0.9 * max([Q_tables.value(new_features, a) for a in range(15)])
                Q_tables.update(features, translated_action, target)
                
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    if reached_road:
                        successes += 1
                    break


            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))
            print('Success Rate: ', successes/(e+1))


class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, features, Q_tables, epsilon):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        if np.random.random() > epsilon:
            Q_vals = [Q_tables.value(features, a) for a in range(15)]
            translated_action = np.argmax(Q_vals)
            # print(Q_vals[translated_action])
            action = num_to_action(translated_action)
        else:
            action = num_to_action(np.random.randint(0, 15))

        return action

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        global weight_file

        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        successes = 0
        ranges = [[0,1000], [0, 10], [0, 360], [0, 1000], [0, 1000], [0, 1000]] # distance to road, velocity, angle, distance to obstacle {l, c, r}
        num_tilings = 4
        epsilon = 0.1
        bin_counts = [10, 3, 20, 3, 3, 3]
        bins = [bin_counts for _ in range(num_tilings)]
        offsets = [[i/num_tilings * bin_counts[j] for j in range(len(bin_counts))] for i in range(num_tilings)]

        tilings = create_tilings(ranges, num_tilings, bins, offsets)
        actions = [a for a in range(15)]
        Q_tables = QValueFunction(tilings, actions, 0.1, weight_file)

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]            
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            print('Episode Num:', e)
            # epsilon = epsilon / ((e/10)+1)
            road_status = False

            for t in range(TIMESTEPS):
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()
                
                features = get_features(state, ran_cen_list)
                action = self.next_action(features, Q_tables, epsilon)
                translated_action = action_to_num(action)
                
                old_state = state
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                
                new_features = get_features(state, ran_cen_list)

                if state[1] > 0:
                    angle_to_road = 360 - math.atan(state[1]/(350-state[0])) * 180 / math.pi
                else:
                    angle_to_road = math.atan(-state[1]/(350-state[0])) * 180 / math.pi

                delta = angle_to_road - state[3]
                if delta < -180: 
                    delta = delta + 360
                elif delta > 180:
                    delta = 360 - delta

                if (new_features[0] < features[0] and new_features[1] > 1):
                    r = 10 * math.cos((delta) * math.pi / 180)
                    reward += r
                    # print('reward', r, angle_to_road)
                else:
                    reward -= new_features[0] - features[0]
        
                if new_features[3] < 100:
                    reward -= (100 - new_features[3])/3
                    # print('OBSTACLE DETECTED LEFT')
                if new_features[4] < 100:
                    reward -= (100 - new_features[4])
                    # print('OBSTACLE DETECTED AHEAD')
                elif new_features[4] == 500 and new_features[1] > 1:
                    # print('POINTING TOWARDS ROAD!')
                    reward += 10
                if new_features[5] < 100:
                    reward -= (100 - new_features[5])/3
                    # print('OBSTACLE DETECTED RIGHT')
            

                # if abs(old_state[3] - state[3]) < 1 and abs(features[0] - new_features[0]) < 1:
                #     reward += -100 
                # else:
                #     if new_features[3]:
                #         reward += -(100 - new_features[4]) / 1.25
                #     # elif new_features[4] == 500:
                #     #     reward += 10
                #     else:
                #         if new_features[4]:
                #             reward += -(100 - new_features[4]) / 20
                #         if new_features[5]:
                #             reward += -(100 - new_features[5]) / 20
                        
                #         if new_features[1] > 1:
                #             reward += 10 * math.cos(delta * math.pi / 180)
                            

                # print(reward)

                target = reward + 0.9 * max([Q_tables.value(new_features, a) for a in range(15)])
                Q_tables.update(features, translated_action, target)

                
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    if reached_road:
                        successes += 1
                    break

            print(str(road_status) + ' ' + str(cur_time))
            print('Success Rate: ', successes/(e+1))


            if not weight_file:
                weight_file = 'weights.npy'
            
            print('Saving weights...')
            Q_tables.save_weights(weight_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
