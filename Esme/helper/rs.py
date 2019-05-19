# Set random seed
import numpy as np
import tensorflow as tf

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
