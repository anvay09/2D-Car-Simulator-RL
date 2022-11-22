from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

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
    
    for t in range(60):
        px, py = get_point(x, y, ndx, ndy, t)

        for centre in mud_pit_centres:
            x1 = centre[0]-50
            x2 = centre[0]+50
            y1 = centre[1]-50
            y2 = centre[1]+50

            if px >= x1 and px <= x2 and py >= y1 and py <= y2:
                # print('MUD PIT DETECTED, t = ', t)
                return 1

        if px > 350 and px < 355 and py < 50 and py > -50: # road
            return 0

        elif px > 350 or px < -350 or py > 350 or py < -350:
            # print('WALL DETECTED, t = ', t)
            return 1
    
    return 0

def action_to_num(action):
    return action[0] * 5 + action[1]
    
def num_to_action(num):
    action = np.array([0,0]) 
    action[1] = num % 5
    action[0] = (num - action[1])/5
    return action

def features_to_state_ind(features, feature_ranges, feature_bin_counts):
    state = 0
    acc = 1

    for f in range(len(features)):
        f_val = features[f]
        f_range = feature_ranges[f]
        f_bin = feature_bin_counts[f]
        bin_array = np.linspace(f_range[0], f_range[1], f_bin)
        ind = np.digitize(f_val, bin_array) - 1
        # print(acc, ind)
        state += ind * acc
        acc *= f_bin

    return state

def get_features(state, mud_pit_centres = []):
    x = state[0]
    y = state[1]
    vel = state[2]
    angle = state[3]

    distance_to_road = np.sqrt((350-x)**2 + (y)**2)
    if angle > 180:
        angle = angle - 360
    
    # distance_to_obstacle_45 = cast_ray(x, y, angle, 45, mud_pit_centres)
    distance_to_obstacle_0 = cast_ray(x, y, angle, 0, mud_pit_centres)
    # distance_to_obstacle_m45 = cast_ray(x, y, angle, -45, mud_pit_centres)
    # distance_to_obstacle_120 = cast_ray(x, y, angle, 120, mud_pit_centres)
    # distance_to_obstacle_m120 = cast_ray(x, y, angle, -120, mud_pit_centres)

    # features = [distance_to_road, vel, angle, distance_to_obstacle_0, distance_to_obstacle_60, distance_to_obstacle_m60, distance_to_obstacle_120, distance_to_obstacle_m120]
    # features = [distance_to_road, distance_to_obstacle_0, vel, angle, distance_to_obstacle_45, distance_to_obstacle_m45]
    # features = [distance_to_road, distance_to_obstacle_0, vel, angle]
    features = [x, y, vel, angle]
    return features

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        # if np.random.random() < epsilon:
        #     steer_action = np.random.randint(0,3)
        #     acc_action = np.random.randint(0,5)
        #     action = np.array([steer_action, acc_action])

        # else:
        #     Q_vals = [Q_table[state_ind][a] for a in range(15)]
        #     # print(Q_vals)
        #     action = num_to_action(np.argmax(Q_vals))
    
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

        # print(delta)
        
        if delta > 3:
            action_steer = 2
            action_acc = 2

        elif delta < -3:
            action_steer = 0
            action_acc = 2

        else:
            action_steer = 1
            action_acc = 4        
        
        action = np.array([action_steer, action_acc])

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
        # bin_widths = [50, 2, 3, 120]
        # feature_ranges = [[0, 1000], [0, 1], [0, 10], [-180, 180]]
        # num_states = np.prod(bin_widths)
        # num_actions = 15
        # epsilon = 0.1
        # learning_rate = 0.1
        # n = 5
        

        # if os.path.isfile(weight_file):
        #     print('Loading saved weights...')
        #     with open(weight_file, 'rb') as f:
        #         Q_table = np.load(f)
        # else:
        #     Q_table = np.zeros((num_states, num_actions))

        # R = np.zeros((num_states, num_actions))
        # T = np.zeros((num_states, num_actions, num_states))
        # print(T.shape)
        # count = np.zeros((num_states, num_actions))
        # observed_states = set()
        # actions_taken_at_S = np.zeros((num_states, num_actions))

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
            # print('Episode Num:', e)
            # if e % 100 == 0:
            #     epsilon = epsilon / 2
            #     learning_rate = learning_rate / 2
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

                # features = get_features(state)
                # old_distance = np.sqrt((350-state[0])**2 + (state[1]**2))
                # old_distance = features[0]
                # state_ind = features_to_state_ind(features, feature_ranges, bin_widths)
                # observed_states.add(state_ind)
                action = self.next_action(state)
                # action_ind = action_to_num(action)
                # actions_taken_at_S[state_ind][action_ind] = 1
                # print(actions_taken_at_S[state_ind])

                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                # new_features = get_features(state)
                

                # if state[1] > 0:
                #     angle_to_road = 360 - math.atan(state[1]/(350-state[0])) * 180 / math.pi
                # else:
                #     angle_to_road = math.atan(-state[1]/(350-state[0])) * 180 / math.pi

                # if angle_to_road > 180:
                #     angle_to_road = 360 - angle_to_road

                # distance = np.sqrt((350-state[0])**2 + (state[1]**2))
                # distance = new_features[0]
                # delta = angle_to_road - new_features[3]
                # if delta < -180: 
                #     delta = delta + 360
                # elif delta > 180:
                #     delta = 360 - delta


                # r = math.cos((delta * math.pi / 180))
                # if distance < old_distance:
                #     reward += 3 * r

                # reward += r

                # if new_features[1] < 500 and features[1] - new_features[1] > 0.5:
                #     reward += - (500 - new_features[1]) / 50
                # elif new_features[1] > 500 and new_features[2] > 2:
                #     reward += 20
                

                # if state[2] == 0:
                #     reward += -10

                # print(reward)

                # new_state_ind = features_to_state_ind(new_features, feature_ranges, bin_widths)
                # T[state_ind][action_ind][new_state_ind] += 1

                # Q_table[state_ind][action_ind] += learning_rate * (reward + 0.9 * max([Q_table[new_state_ind][a] for a in range(15)]) - Q_table[state_ind][action_ind])
                
                # avg_reward = R[state_ind][action_ind]
                # num = count[state_ind][action_ind]
                # R[state_ind][action_ind] = (avg_reward * num + reward) / (num + 1)
                # count[state_ind][action_ind] += 1

                # for _ in range(n):
                #     S = random.choice(tuple(observed_states))
                #     previously_taken_actions = []
                #     for a in range(15):
                #         if actions_taken_at_S[S][a]:
                #             previously_taken_actions += [a]
                #     A = random.choice(previously_taken_actions)

                #     state_transitions = [T[S][A][s] for s in range(num_states)]
                #     num = sum(state_transitions)
                #     distribution = [state_transitions[s] / (num) for s in range(len(state_transitions))]
                #     S_dash = int(np.random.choice(range(num_states), p = distribution))
                    # S_dash = np.argmax(state_transitions)
                    
                    # reward = R[S][A]

                    # Q_table[S][A] += learning_rate * (reward + 0.9 * max([Q_table[S_dash][a] for a in range(15)]) - Q_table[S][A])



                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    if road_status:
                        successes += 1
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))
            # print('Success Rate:', successes/(e+1))

            # with open(weight_file, 'wb') as f:
            #     print('Saving weights...')
            #     np.save(f, Q_table)

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
                    # print('Case 10', angle)
                    # if (angle > 180):
                    action = np.array([2, 2])
                    # else:
                    #     action = np.array([0, 2])
                
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

        successes = 0
        # bin_widths = [20, 2, 3, 30]
        # num_states = np.prod(bin_widths)
        # num_actions = 15
        # epsilon = 0.1
        # learning_rate = 0.1
        # n = 20
        # feature_ranges = [[0, 1000], [0, 1000], [0, 10], [-180, 180]]

        # if os.path.isfile(weight_file):
        #     print('Loading saved weights...')
        #     with open(weight_file, 'rb') as f:
        #         Q_table = np.load(f)
        # else:
        #     Q_table = np.zeros((num_states, num_actions))

        # R = np.zeros((num_states, num_actions))
        # T = np.zeros((num_states, num_actions, num_states))
        # # print(T.shape)
        # count = np.zeros((num_states, num_actions))
        # observed_states = set()
        # actions_taken_at_S = np.zeros((num_states, num_actions))

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
            # print('Episode Num:', e)
            # if e % 100 == 0:
            #     epsilon = epsilon / 2
            #     learning_rate = learning_rate / 2
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

                # features = get_features(state)
                # # old_distance = np.sqrt((350-state[0])**2 + (state[1]**2))
                # old_distance = features[0]
                # state_ind = features_to_state_ind(features, feature_ranges, bin_widths)
                # observed_states.add(state_ind)
                # action = self.next_action(state_ind, Q_table, epsilon)
                # action_ind = action_to_num(action)
                # actions_taken_at_S[state_ind][action] = 1

                action = self.next_action(state, ran_cen_list)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                # new_features = get_features(state)

                # if state[1] > 0:
                #     angle_to_road = 360 - math.atan(state[1]/(350-state[0])) * 180 / math.pi
                # else:
                #     angle_to_road = math.atan(-state[1]/(350-state[0])) * 180 / math.pi

                # if angle_to_road > 180:
                #     angle_to_road = 360 - angle_to_road

                # distance = np.sqrt((350-state[0])**2 + (state[1]**2))
                # distance = new_features[0]
                # delta = angle_to_road - new_features[3]
                # if delta < -180: 
                #     delta = delta + 360
                # elif delta > 180:
                #     delta = 360 - delta

                # r = 10 * math.cos((delta) * math.pi / 180) * math.cos((delta) * math.pi / 180)
                # if distance < old_distance and new_features[2] > 1:
                #     reward += r

                # if new_features[1] < 500:
                #     reward += - (500 - new_features[1]) / 25
                # elif new_features[1] > 500 and new_features[2] > 2:
                #     reward += 20
                # if new_features[0] > 320 and (new_features[1] > 100 or new_features[1] < -100):
                #     reward += -10
                # elif new_features[1] > 320 or new_features[1] < -320:
                #     reward += -10
                # elif new_features[0] < -320:
                #     reward += -10

                # if state[2] == 0:
                #     reward += -10

                # new_state_ind = features_to_state_ind(new_features, feature_ranges, bin_widths)
                # T[state_ind][action_ind][new_state_ind] += 1
                # Q_table[state_ind][action_ind] += learning_rate * (reward + 0.9 * max([Q_table[new_state_ind][a] for a in range(15)]) - Q_table[state_ind][action_ind])
                
                # avg_reward = R[state_ind][action_ind]
                # num = count[state_ind][action_ind]
                # R[state_ind][action_ind] = (avg_reward * num + reward) / (num + 1)
                # count[state_ind][action_ind] += 1

                # for _ in range(n):
                #     S = random.choice(tuple(observed_states))
                #     previously_taken_actions = []
                #     for a in range(15):
                #         if actions_taken_at_S[S][a]:
                #             previously_taken_actions += [a]
                #     A = random.choice(previously_taken_actions)

                #     S_dash = np.argmax([T[S][A][s] for s in range(num_states)])
                #     reward = R[S][A]

                #     Q_table[S][A] += learning_rate * (reward + 0.9 * max([Q_table[S_dash][a] for a in range(15)]) - Q_table[S][A])

                
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    if road_status:
                        successes += 1
                    break

            print(str(road_status) + ' ' + str(cur_time))
            # print('Success Rate:', successes/(e+1))

            # with open(weight_file, 'wb') as f:
            #     print('Saving weights...')
            #     np.save(f, Q_table)

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
