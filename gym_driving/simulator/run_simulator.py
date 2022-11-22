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
        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
            # print('Episode Num:', e)
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

                
                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
            
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

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