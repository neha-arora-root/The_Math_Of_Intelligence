import numpy as np
import math
from PIL import Image

class SelfOrganizingMaps(object):
    
    '''
    Kohonen's Self Organizing Maps
    '''
    
    def __init__(self, output_dim, input_dim, t_step, num_iters=1000, dtype=np.float32):
        
        '''
        Initialize the network weights
        '''
        self.width, self.height = output_dim
        self.weights = np.random.randn(self.width, self.height, input_dim)
        self.num_iters = num_iters
        self.map_radius = max(self.weights.shape)/2
        self.lambda_const = self.num_iters/math.log(self.map_radius)
        self.t_step = t_step
        
    
    def get_bmu(self, x):
        distance = np.sum((self.weights - x) ** 2, 2)
        min_idx = distance.argmin()
        return np.unravel_index(min_idx, distance.shape)
        
    def get_bmu_dist(self, train):
        # initialize array where values are its index
        x, y, other_dims = self.weights.shape
        xi = np.arange(x).reshape(x, 1).repeat(y, 1)
        yi = np.arange(y).reshape(1, y).repeat(x, 0)
        # returns matrix of distance of each index in 2D from BMU
        return np.sum((np.dstack((xi, yi)) - np.array(self.get_bmu(train))) ** 2, 2)
    
    
    def get_neighborhood_radius(self, iter_idx):
        return self.map_radius * np.exp(-iter_idx/self.lambda_const)
        
    
    def train_row(self, train, iter_idx, learning_rate):
        neighborhood_radius = self.get_neighborhood_radius(iter_idx)
        bmu_dist = self.get_bmu_dist(train).astype('float64')
        
        # exponential decaying learning rate
        lr = learning_rate * np.exp(-iter_idx/self.num_iters) 
        
        # influence
        theta = np.exp(-(bmu_dist)/ (2 * neighborhood_radius ** 2))
        return np.expand_dims(theta, 2) * (train - self.weights)
    
    def train(self, train_set, learning_rate=1e-3):
        
        for i in range(self.num_iters):
            
            if i%2 == 0:
                print("Training iteration: ", i)
                
            for j in range(len(train_set)):
                self.weights += self.train_row(train_set[j], i, learning_rate)         
                
                

    def show(self):
        im = Image.fromarray(self.weights.astype('uint8'), mode='RGB')
        im.format = 'JPG'
        im.show()      
        return
    
    
    #######################################################################################