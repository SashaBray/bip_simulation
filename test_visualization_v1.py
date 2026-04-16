
import time
import cascadio
import pickle
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from colider_simulation_instruments_lib import *


configuration_directory_link = 'C:\\Users\\Sasha Bray\\Desktop\\project\\database\\'
configuration_id = 3

configuration = Collider_configuration(configuration_directory_link, 
                                       configuration_id) # Класс, содержащий информацию о конфигурации установки


visual = Colider_simulator_visualization(configuration)


# 2. Загружаем (Десериализация)
with open('out_simulation_0.pkl', 'rb') as file:
    out = pickle.load(file)
print('out', out.y2.shape)


visual.trajectories = out.y2
visual.show_all()







