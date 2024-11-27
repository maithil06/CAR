import pandas as pd
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_folder = "D:/MLis-Car/machine-learning-in-science-ii-2024/training_data/training_data/"
test_folder = "D:/MLis-Car/machine-learning-in-science-ii-2024/test_data/test_data"
train = pd.read_csv('machine-learning-in-science-ii-2024/training_norm.csv')