import math

state = [-150, -20, 0.5, 275]

if state[1] > 0:
    required_angle = 360 - math.atan(state[1]/(350-state[0])) * 180 / math.pi
else:
    required_angle = math.atan(-state[1]/(350-state[0])) * 180 / math.pi

delta = required_angle - state[3]

if delta < -180: 
    delta = delta + 360
elif delta > 180:
    delta = 360 - delta


reward = 10 * math.cos(delta * math.pi / 180)
print(delta, reward)