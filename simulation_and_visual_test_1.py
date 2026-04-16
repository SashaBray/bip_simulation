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

coordinates = create_many_pionts(5, 5, 0.02, 0)
# print('coordinates', coordinates)
# exit()

С = 299792458       # Скорость света 
persent = 0.80      # Скорость частиц в процентах от скорости света 
Q = -1.60217653e-19     # Заряд электрона 
M_e = 9.109383713928e-31    # Масса покоя электрона 

speeds = np.zeros(coordinates.shape)    
speeds[:, 0] = 1 * С * persent  

masses_rest = M_e * np.ones((coordinates.shape[0], 1), dtype='float64') 
charges = Q * np.ones((coordinates.shape[0], 1), dtype='float64') 


    # Инициализация виртуальной реальности и установки 
    
configuration_directory_link = 'C:\\Users\\Sasha Bray\\Desktop\\project\\database\\'
configuration_id = 8

configuration = Collider_configuration(configuration_directory_link, 
                                       configuration_id) # Класс, содержащий информацию о конфигурации установки

reality = Colider_simulator()
all_fields = Field_generation_device() # Класс, соджержащий аппроксимации всех магнитных и электрических устройств. 
all_fields.add_configuration(configuration) # Передавем классу с апроксимациями класс конфигурации. Запускается процедура инициализации оберток полей.

reality.fields_func = all_fields

time_sim = 10/С
       
# delta_time = 
       
       
    # Запустим симуляцию  
        
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


# 1. Сохраняем (Сериализация)
with open('out_simulation_1.pkl', 'wb') as file:
    pickle.dump(out, file)



    # Визуализируем результаты 
    
with open('out_simulation_1.pkl', 'rb') as file:
    out = pickle.load(file)

visual = Colider_simulator_visualization(configuration)

visual.trajectories = out.y2
visual.show_all()

