from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *
from tiles3 import IHT, tiles

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 2000

def get_normalized_direction(angle):
    dx = math.cos(angle * math.pi / 180)
    dy = math.sin(angle * math.pi / 180)
    return dx, dy

def get_point(x, y, dx, dy, t):
    px = x + t*dx
    py = y + t*dy
    return px, py

def cast_ray(x, y, angle, theta, mud_pit_centres = [], threshold = 1000):
    dx, dy = get_normalized_direction(angle)

    ndx = dx * math.cos(theta * math.pi / 180) - dy * math.sin(theta * math.pi / 180)
    ndy = dx * math.sin(theta * math.pi / 180) + dy * math.cos(theta * math.pi / 180)
    
    # threshold = 1000

    for t in range(threshold):
        px, py = get_point(x, y, ndx, ndy, t)

        for centre in mud_pit_centres:
            x1 = centre[0]-50
            x2 = centre[0]+50
            y1 = centre[1]-50
            y2 = centre[1]+50

            if px >= x1 and px <= x2 and py >= y1 and py <= y2:
                val = - (threshold - t)/100
                # if theta == 0:
                    # print('MUD PIT DETECTED', val)
                return val
                # return -1

        if px > 350 and px < 355 and py < 50 and py > -50: # road
            return (threshold - t)/100
            # return 1

        elif px > 350 or px < -350 or py > 350 or py < -350:
            # print('WALL DETECTED, t = ', t)
            return - (threshold - t)/100
            # return -1
    
    return 0
    
def cast_thick_ray(x, y, angle, theta, mud_pit_centres = [], threshold = 1000):
    l = cast_ray(x, y, angle + 1, theta, mud_pit_centres, threshold)
    r = cast_ray(x, y, angle - 1, theta, mud_pit_centres, threshold)

    return min(l, r)

def action_to_num(action):
    return action[0] * 5 + action[1]
    
def num_to_action(num):
    action = np.array([0,0]) 
    action[1] = num % 5
    action[0] = (num - action[1]) / 5
    return action

def Q(w, x):
    return np.sum(w[x])

def mytiles(features, feature_ranges, iht, numTilings, numTiles, action):
    scaleFactor = numTiles / (feature_ranges[1] - feature_ranges[0])
    return tiles(iht, numTilings, [features * scaleFactor], [action])


def get_features(state, threshold = 1000, mud_pit_centres = []):
    x = state[0]
    y = state[1]
    vel = state[2]
    angle = state[3]

    if state[1] > 0:
        required_angle = 360 - math.atan(state[1]/(350-state[0])) * 180 / math.pi
    else:
        required_angle = math.atan(-state[1]/(350-state[0])) * 180 / math.pi
    
    delta = required_angle - state[3]

    if delta < -180: 
        delta = delta + 360
    elif delta > 180:
        delta = 360 - delta

    # distance_to_road = np.sqrt((350-x)**2 + (y)**2)
    d_0 = cast_thick_ray(x, y, angle, 0, mud_pit_centres, threshold)
    d_45 = cast_thick_ray(x, y, angle, 45, mud_pit_centres, threshold)
    d_90 = cast_thick_ray(x, y, angle, 90, mud_pit_centres, threshold)
    d_m45 = cast_thick_ray(x, y, angle, -45, mud_pit_centres, threshold)
    d_m90 = cast_thick_ray(x, y, angle, -90, mud_pit_centres, threshold)
    # d_135 = cast_ray(x, y, angle, 135, mud_pit_centres)
    # d_m135 = cast_ray(x, y, angle, -135, mud_pit_centres)
    # d_180 = cast_ray(x, y, angle, 180, mud_pit_centres)
    
    # features = [distance_to_obstacle, distance_to_obstacle_l, distance_to_obstacle_r, distance_to_road, vel, angle]
    # features = [x, y, vel, angle]
    features = [d_0, d_45, d_90, d_m45, d_m90, vel, delta]

    return features

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, features, feature_ranges, ihts, num_tilings, num_tiles, num_actions, weights, epsilon):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """    
        if np.random.random() > epsilon:
            Q_vals = np.zeros(num_actions)
            for a in range(num_actions):
                for i in range(len(features)):
                    x = mytiles(features[i], feature_ranges[i], ihts[i], num_tilings[i], num_tiles[i], a)
                    Q_vals[a] += Q(weights[i], x)

            # a = np.argmax(Q_vals)
            a = np.random.choice(np.flatnonzero(Q_vals == Q_vals.max()))
            action = num_to_action(a)
            
            # print(action, Q_vals[a])
        else:
            action = num_to_action(np.random.randint(0, 15))
        return action 

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################
        weight_file = 'weights_T1.npy'
        successes = 0
        num_features = 4
        num_actions = 15
        # feature_ranges = [[-350, 350], [-350, 350], [0, 10], [0, 360]]
        # num_tiles = [100, 100, 10, 36]
        # num_tilings = [64, 64, 8, 64]
        # feature_ranges = [[-5, 5], [-5, 5], [-5, 5], [0, 1000], [0, 10], [0, 360]]
        # num_tiles = [10, 10, 10, 10, 10, 20]
        # num_tilings = [8, 8, 8, 8, 8, 8]

        feature_ranges = [[-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10], [0, 10], [-180, 180]]
        num_tiles = [7, 7, 7, 7, 7, 7, 20]
        num_tilings = [8, 8, 8, 8, 8, 8, 8]
        learning_rate = 0.5 / 8
        gamma = 0.9
        epsilon = 0.1
        lam = 0.9
        # maxsize = [200000, 200000, 2000, 50000]
        maxsize = [1000] * 6 + [4000]
        ihts = [IHT(maxsize[i]) for i in range(len(maxsize))]
        weights = [np.zeros(maxsize[i]) for i in range(len(maxsize))]

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
            print('Episode Num:', e)
            z = [np.zeros(maxsize[i]) for i in range(len(maxsize))]
            
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                old_features = get_features(state)
                old_distance = np.sqrt((state[0] - 350)**2 + state[1]**2)
                action = self.next_action(old_features, feature_ranges, ihts, num_tilings, num_tiles, num_actions, weights, epsilon)
                action_ind = action_to_num(action)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                features = get_features(state)
                distance = np.sqrt((state[0] - 350)**2 + state[1]**2)

                # modify reward here #
                ###
                # if state[1] > 0:
                #     required_angle = 360 - math.atan(state[1]/(350-state[0])) * 180 / math.pi
                # else:
                #     required_angle = math.atan(-state[1]/(350-state[0])) * 180 / math.pi
                
                # delta = required_angle - state[3]
        
                # if delta < -180: 
                #     delta = delta + 360
                # elif delta > 180:
                #     delta = 360 - delta

                # if distance < old_distance:
                #     reward += 5 * math.cos(delta * math.pi / 180) * math.cos(delta * math.pi / 180)
                # else:
                #     reward -= 5 * math.cos(delta * math.pi / 180) * math.cos(delta * math.pi / 180)

                # print(reward)

                # if features[1] < old_features[1]:
                #     reward += -10 * state[2]

                # if features[1] == 200:
                #     reward += 20 * state[2]
                #     print('POINTING TOWARDS ROAD', reward)
                
                if not terminate:
                    if old_distance - distance > 0.5:
                        if features[0] > 0:
                            reward = features[0] * (1 + features[5])
                            # print('MOVING TOWARDS ROAD', reward)
                        elif features[0] < 0:
                            reward = -1 + features[0] * (1 + features[5])
                        else:
                            reward = 1
                    else:
                        if features[0] < 0:
                            reward = -1 + features[0] * (1 + features[5])
                            # print('MOVING TOWARDS WALL', reward)
                        else:
                            reward = -1


                    if abs(old_features[6]) - abs(features[6]) > 0.5:
                        reward += (1.8 / abs(features[6]))
                        # print('TURNING TOWARDS ROAD', reward)
                    elif abs(old_features[6]) - abs(features[6]) < -0.5:
                        # if abs(features[6]) > 10:
                        reward -= (abs(features[6]) / 1.8)

                    # print(reward)
                ###

                delta = [reward] * num_features
                
                for i in range(num_features):
                    x_t = mytiles(old_features[i], feature_ranges[i], ihts[i], num_tilings[i], num_tiles[i], action_ind)
                    
                    for ind in x_t:
                        delta[i] = delta[i] - weights[i][ind]
                        z[i][ind] = 1
                
                fpsClock.tick(FPS)
                cur_time += 1

                if terminate:
                    road_status = reached_road

                    if road_status:
                        successes += 1

                    for i in range(num_features):
                        weights[i] += learning_rate * delta[i] * z[i]
                        
                    break
                else:
                    next_action = self.next_action(features, feature_ranges, ihts, num_tilings, num_tiles, num_actions, weights, epsilon)
                    next_action_ind = action_to_num(next_action)

                    for i in range(num_features):
                        x_t1 = mytiles(features[i], feature_ranges[i], ihts[i], num_tilings[i], num_tiles[i], next_action_ind)
                        
                        for ind in x_t1:
                            delta[i] = delta[i] + gamma * weights[i][ind]

                        weights[i] += learning_rate * delta[i] * z[i]
                        z[i] = gamma * lam * z[i]
                    
            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))
            print('Success Rate:', successes/(e+1))
            # TC.save_weights(weight_file)
            # print('Saving weights...')
            # np.save(weight_file, weights)

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, features, feature_ranges, ihts, num_tilings, num_tiles, num_actions, weights, epsilon):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """
        if np.random.random() > epsilon:
            Q_vals = np.zeros(num_actions)
            for a in range(num_actions):
                for i in range(len(features)):
                    x = mytiles(features[i], feature_ranges[i], ihts[i], num_tilings[i], num_tiles[i], a)
                    Q_vals[a] += Q(weights[i], x)

            # a = np.argmax(Q_vals)
            a = np.random.choice(np.flatnonzero(Q_vals == Q_vals.max()))
            action = num_to_action(a)
            
            # print(action, Q_vals[a])
        else:
            action = num_to_action(np.random.randint(0, 15))
        return action 
        

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################
        weight_file = 'weights_T2.npy'

        successes = 0
        num_features = 4
        num_actions = 15

        feature_ranges = [[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [0, 10], [-180, 180]]
        num_tiles = [7, 7, 7, 7, 7, 7, 20]
        num_tilings = [8, 8, 8, 8, 8, 8, 8]
        learning_rate = 0.5 / 8
        gamma = 0.97
        epsilon = 0.1
        lam = 0.9
        # maxsize = [200000, 200000, 2000, 50000]
        maxsize = [1000] * 6 + [4000]
        ihts = [IHT(maxsize[i]) for i in range(len(maxsize))]
        weights = [np.zeros(maxsize[i]) for i in range(len(maxsize))]

        for e in range(NUM_EPISODES):
            print('Episode Num:', e)
            z = [np.zeros(maxsize[i]) for i in range(len(maxsize))]
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
            road_status = False

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                old_features = get_features(state, 200, ran_cen_list)
                old_distance = np.sqrt((state[0] - 350)**2 + state[1]**2)
                action = self.next_action(old_features, feature_ranges, ihts, num_tilings, num_tiles, num_actions, weights, epsilon)
                action_ind = action_to_num(action)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                features = get_features(state, 200, ran_cen_list)
                distance = np.sqrt((state[0] - 350)**2 + state[1]**2)

                if not terminate:
                    if old_distance - distance > 0.5:
                        if features[0] > 0:
                            reward = features[0] * (1 + features[5])
                            # print('MOVING TOWARDS ROAD', reward)
                        elif features[0] < 0:
                            reward = -1 + 5 * features[0] * (1 + features[5])
                        else:
                            reward = 1
                    else:
                        if features[0] < 0:
                            reward = -1 + 5 * features[0] * (1 + features[5])
                            # print('MOVING TOWARDS WALL', reward)
                        else:
                            reward = -1


                    if abs(old_features[6]) - abs(features[6]) > 0.5:
                        reward += (1.8 / abs(features[6]))
                        # print('TURNING TOWARDS ROAD', reward)
                    elif abs(old_features[6]) - abs(features[6]) < -0.5:
                        reward -= (abs(features[6]) / 18)
                        # print('TURNING TOWARDS WALL', reward)


                delta = [reward] * num_features
                
                for i in range(num_features):
                    x_t = mytiles(old_features[i], feature_ranges[i], ihts[i], num_tilings[i], num_tiles[i], action_ind)
                    
                    for ind in x_t:
                        delta[i] = delta[i] - weights[i][ind]
                        z[i][ind] = 1
                
                fpsClock.tick(FPS)
                cur_time += 1

                if terminate:
                    road_status = reached_road

                    if road_status:
                        successes += 1

                    for i in range(num_features):
                        weights[i] += learning_rate * delta[i] * z[i]
                        
                    break
                else:
                    next_action = self.next_action(features, feature_ranges, ihts, num_tilings, num_tiles, num_actions, weights, epsilon)
                    next_action_ind = action_to_num(next_action)

                    for i in range(num_features):
                        x_t1 = mytiles(features[i], feature_ranges[i], ihts[i], num_tilings[i], num_tiles[i], next_action_ind)
                        
                        for ind in x_t1:
                            delta[i] = delta[i] + gamma * weights[i][ind]

                        weights[i] += learning_rate * delta[i] * z[i]
                        z[i] = gamma * lam * z[i]



            print(str(road_status) + ' ' + str(cur_time))
            print('Success Rate:', successes/(e+1))

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
