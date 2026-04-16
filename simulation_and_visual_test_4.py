import time
import cascadio
import pickle
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from colider_simulation_instruments_lib import *



    # Инициализируем частицы и их параметры

coordinates = create_many_pionts(5, 5, 0.008, 0)
print('coordinates', coordinates)
# exit()

С = 299792458       # Скорость света 
persent = 0.90      # Скорость частиц в процентах от скорости света 
Q = -1.60217653e-19     # Заряд электрона 
M_e = 9.109383713928e-31    # Масса покоя электрона 

speeds = np.zeros(coordinates.shape)    
speeds[:, 0] = 1 * С * persent  

masses_rest = M_e * np.ones((coordinates.shape[0], 1), dtype='float64') 
charges = Q * np.ones((coordinates.shape[0], 1), dtype='float64') 


    # Инициализация виртуальной реальности и установки 
    
configuration_directory_link = 'C:\\Users\\Sasha Bray\\Desktop\\project\\database\\'.replace('\\', os.sep)
configuration_id = 10

configuration = Collider_configuration(configuration_directory_link, 
                                       configuration_id) # Класс, содержащий информацию о конфигурации установки

reality = Colider_simulator()
all_fields = Field_generation_device() # Класс, соджержащий аппроксимации всех магнитных и электрических устройств. 
all_fields.add_configuration(configuration) # Передавем классу с апроксимациями класс конфигурации. Запускается процедура инициализации оберток полей.

reality.fields_func = all_fields




# visual = Colider_simulator_visualization(configuration)
# visual.show_all()




S_traectory = 5
time_sim = S_traectory / (С * persent)
       


delta_array = np.ones_like(configuration.state_arguments_vector)

steps = np.array([1, 1, 1]) * 0.01
lerning_rate = 1
n = 5

y_i_array = []

# print('all_fields', all_fields)

new_arg_vector = np.array([1, 1, 1]) * 50




# configuration.state_arguments_vector = new_arg_vector




    
start_time = time.time()  # Фиксируем время начала

out = reality.simulate(speeds, 
                coordinates,
                charges,
                masses_rest,
                time_sim,
                #  delta_time=
                )

end_time = time.time()    # Фиксируем время конца
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time} секунд")

y_i = beam_spread(out.y2)
    
    

# 1. Сохраняем (Сериализация)
with open('out_simulation_1.pkl', 'wb') as file:
    pickle.dump(out, file)





    # Визуализируем результаты 
        
with open('out_simulation_1.pkl', 'rb') as file:
    out = pickle.load(file)

visual = Colider_simulator_visualization(configuration)

visual.trajectories = out.y2
visual.show_all()


print()


