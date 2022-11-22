import numpy as np
import os

class TileCoding():
    
    def __init__(self, k, a):
        self.num_features = k
        self.num_actions = a
        self.weights = np.array([[None for _ in range(a)] for _ in range(k)])
        self.tilings = np.array([None for _ in range(k)])

    def create_tilings_for_feature(self, feature_index, feature_range, num_tilings, num_tiles):
        tilings_for_feature = np.array([None for _ in range(num_tilings)])

        for a in range(self.num_actions):
            weights_for_feature = np.zeros(num_tiles * num_tilings)
            self.weights[feature_index][a] = weights_for_feature

        offset_multiplier = 1 / num_tilings

        for m in range(num_tilings):
            offset = m * offset_multiplier
            tiling = np.linspace(feature_range[0], feature_range[1], num_tiles + 1)[1:-1] + offset
            tilings_for_feature[m] = tiling
            
        self.tilings[feature_index] = tilings_for_feature

    def get_feature_vector(self, feature_index, feature_value):
        tilings_for_feature = self.tilings[feature_index]
        num_tilings = len(tilings_for_feature)
        num_tiles = len(tilings_for_feature[0]) + 1
        
        x = np.zeros(num_tiles * num_tilings)

        for m in range(num_tilings):
            tiling = tilings_for_feature[m]
            ind = np.digitize(feature_value, tiling)
            x[ind + m * num_tiles] = 1

        return x

    def get_weight_vector(self, feature_index, action):
        w = self.weights[feature_index][action]
        return w

    def update_weights(self, w, action):
        start = 0
        for i in range(self.num_features):
            num_tilings = len(self.tilings[i])
            num_tiles = len(self.tilings[i][0]) + 1
            end = num_tilings * num_tiles + start
            
            w_f = w[start:end]
            self.weights[i][action] = w_f

            start = end
            

    def Q_value(self, features, action): # returns Q Value as well as gradient
        x = np.array([])
        w = np.array([])

        for i in range(self.num_features):
            x_n = self.get_feature_vector(i, features[i])
            w_n = self.get_weight_vector(i, action)

            # print(i, x_n.shape, w_n.shape)
            x = np.concatenate((x, x_n))
            w = np.concatenate((w, w_n))

        Q = np.dot(w, x)

        return Q, w, x

    def load_weights(self, weight_file):
        if os.path.isfile(weight_file):
            print('Loading saved weights...')
            with open(weight_file, 'rb') as f:
                self.weights = np.load(f, allow_pickle = True)

    def save_weights(self, weight_file):
        print('Saving weights...')
        with open(weight_file, 'wb') as f:
            np.save(f, self.weights, allow_pickle = True)

# if __name__ == '__main__':
#     tc = TileCoding(3, 15)
#     tc.create_tilings_for_feature(0, [0, 5], 3, 3)
#     tc.create_tilings_for_feature(1, [0, 5], 3, 5)
#     tc.create_tilings_for_feature(2, [-10, 10], 3, 10)
#     Q, x = tc.Q_value([3.2, 1.3, -1], 7)
#     print(Q)
#     print(x)
