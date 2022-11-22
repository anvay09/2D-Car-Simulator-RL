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
NUM_EPISODES = 1000

weight_file = 'weights_T1.npy'

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

        if px > 350 and px < 355 and py < 50 and py > -50: # road
            return 200

        elif px > 350 or px < -350 or py > 350 or py < -350:
            # print('WALL DETECTED, t = ', t)
            return t
    
    return 150

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
    # scaleFactors = [numTiles[i] / (feature_ranges[i][1] - feature_ranges[i][0]) for i in range(len(features))]

    # return tiles(iht, numTilings, [features[i] * scaleFactors[i] for i in range(len(features))], [action])
    scaleFactor = numTiles / (feature_ranges[1] - feature_ranges[0])
    return tiles(iht, numTilings, [features * scaleFactor], [action])


def get_features(state, mud_pit_centres = []):
    x = state[0]
    y = state[1]
    vel = state[2]
    angle = state[3]

    # distance_to_road = np.sqrt((350-x)**2 + (y)**2)
    # distance_to_obstacle = cast_ray(x, y, angle, 0, mud_pit_centres)
    
    # features = [distance_to_road, distance_to_obstacle, vel, angle]
    features = [x, y, vel, angle]

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
            # Q_vals = [TC.Q_value(features, a)[0] for a in range(15)]
            # a = np.argmax(Q_vals)

            Q_vals = np.zeros(num_actions)
            for a in range(num_actions):
                for i in range(len(features)):
                    x = mytiles(features[i], feature_ranges[i], ihts[i], num_tilings[i], num_tiles[i], a)
                    Q_vals[a] += Q(weights[i], x)

            a = np.argmax(Q_vals)
            # a = np.random.choice(np.flatnonzero(Q_vals == Q_vals.max()))
            action = num_to_action(a)
            # if features[2] == 0.0:
            #     print(action, Q_vals[a])
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
        successes = 0
        num_features = 4
        num_actions = 15
        feature_ranges = [[-350, 350], [-350, 350], [0, 10], [0, 360]]
        num_tiles = [10, 10, 10, 36]
        num_tilings = [256, 256, 8, 64]
        learning_rate = 0.001
        gamma = 1
        epsilon = 0.1
        lam = 0.9
        maxsize = [60000, 60000, 4000, 60000]
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

                # if distance < old_distance and state[2] > 2:
                #     reward += 10 * math.cos(delta * math.pi / 180) * math.cos(delta * math.pi / 180)

                # if features[1] < old_features[1]:
                #     reward += -10 * state[2]

                # if features[1] == 200:
                #     reward += 20 * state[2]
                #     print('POINTING TOWARDS ROAD', reward)

                if distance < old_distance:
                    reward = 1
                else:
                    reward = -1
                ###

                delta = reward
                next_action = self.next_action(features, feature_ranges, ihts, num_tilings, num_tiles, num_actions, weights, epsilon)
                next_action_ind = action_to_num(next_action)

                Q_t = 0
                Q_tp1 = 0
                x = []
                x_next = []

                for i in range(num_features):
                    x_t = mytiles(old_features[i], feature_ranges[i], ihts[i], num_tilings[i], num_tiles[i], action_ind)
                    x_t1 = mytiles(features[i], feature_ranges[i], ihts[i], num_tilings[i], num_tiles[i], next_action_ind)
                
                    Q_t += Q(weights[i], x_t)
                    Q_tp1 += Q(weights[i], x_t1)
    
                    x += [x_t]
                    x_next += [x_t1]

                    for ind in x_t:
                        delta -= weights[i][ind]
                        z[i][ind] += 1
                
                fpsClock.tick(FPS)
                cur_time += 1

                if terminate:
                    road_status = reached_road

                    if road_status:
                        successes += 1

                    for i in range(num_features):
                        weights[i] += learning_rate * delta * z[i]
                        
                    break
                else:
                    for i in range(num_features):
                        x_t1 = x_next[i]

                        for ind in x_t1:
                            delta += gamma * weights[i][ind]

                        weights[i] += learning_rate * delta * z[i]
                        z[i] = gamma * lam * z[i]
                    
            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))
            print('Success Rate:', successes/(e+1))
            # TC.save_weights(weight_file)
            print('Saving weights...')
            np.save(weight_file, weights)

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, state, mud_pit_centres = []):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """
        x = state[0]
        y = state[1]
        vel = state[2]
        angle = state[3]

        if y > 0:
            angle_to_road = 360 - math.atan(y/(350-x)) * 180 / math.pi
        else:
            angle_to_road = math.atan(-y/(350-x)) * 180 / math.pi

        delta = angle_to_road - angle
        if delta < -180: 
            delta = delta + 360
        elif delta > 180:
            delta = 360 - delta

        # Replace with your implementation to determine actions to be taken
        
        if y > 20:
            if cast_ray(x, y, 270, 0, mud_pit_centres) or cast_ray(x, y, 270, 25, mud_pit_centres) or cast_ray(x, y, 270, -25, mud_pit_centres):
                if ((angle > 350 and angle <= 360) or (angle >= 0 and angle < 10)):
                    # print('Case 2')
                    action = np.array([1, 4])
                else:
                    # print('Case 1')
                    if (angle > 180 and angle <= 350):
                        action = np.array([2, 2])
                    else:
                        action = np.array([0, 2])
            else:
                if (angle > 265 and angle < 275):
                    # print('Case 4')
                    action = np.array([1, 4])
                else:
                    # print('Case 3')
                    if (angle > 90 and angle <= 265):
                        action = np.array([2, 2])
                    else:
                        action = np.array([0, 2])
                    
        elif y < -20:
            if cast_ray(x, y, 90, 0, mud_pit_centres) or cast_ray(x, y, 90, 25, mud_pit_centres) or cast_ray(x, y, 90, -25, mud_pit_centres):
                if ((angle > 350 and angle <= 360) or (angle >= 0 and angle < 10)):
                    # print('Case 6')
                    action = np.array([1, 4])
                else:
                    # print('Case 5')
                    if (angle > 180 and angle <= 350):
                        action = np.array([2, 2])
                    else:
                        action = np.array([0, 2])
            else:
                if (angle > 85 and angle < 95):
                    # print('Case 8')
                    action = np.array([1, 4])
                else:
                    if (angle > 270 or angle <= 85):
                    # print('Case 7')
                        action = np.array([2, 2])
                    else:
                        action = np.array([0, 2])

        else:
            if (delta > -5 and delta < 5):
                # print('Case 11')
                action = np.array([1, 4])
            else:
                if vel > 0:
                    # print('Case 9')
                    action = np.array([1, 0])
                else:
                    # print('Case 10')
                    action = np.array([2, 2])
                    
                
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
            road_status = False

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state, ran_cen_list)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            print(str(road_status) + ' ' + str(cur_time))

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
