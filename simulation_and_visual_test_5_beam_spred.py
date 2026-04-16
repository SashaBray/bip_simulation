import time
import cascadio
import pickle
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from colider_simulation_instruments_lib import *

"""Эксперемент по настройке токов магнитных линз для фокусировки пучка с помощью градиентного спуска"""



    # Инициализируем частицы и их параметры

coordinates = create_many_pionts(2, 5, 0.008, 0, dz=0.0, dy=0.0)
# print('coordinates', coordinates)
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
    
configuration_directory_link = 'C:\\Users\\Sasha Bray\\Desktop\\project\\database\\'
configuration_id = 8

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
       
       

coords_init = np.array([-0.5, 0.5])
coords_final = np.array([0.1, -0.1])
steps = 30
 
 
coords_step = (coords_final - coords_init)/steps



args_story = []
delta_array_i = []
beam_story = []
y_max = 100


for i in range(steps+1):
     
    args_i = coords_step * i + coords_init 
    
    configuration.state_arguments_vector = args_i


    out = reality.simulate(speeds,      # Производим моделирование 
                    coordinates,
                    charges,
                    masses_rest,
                    time_sim,
                    #  delta_time=
                    )


    y_j = beam_spread(out.y2)               # Измеряем разлет частиц
    print('\n', 'итерация', i)       
    print('beam_spread', y_j, 'configuration.state_arguments_vector', configuration.state_arguments_vector)

    
    
    if y_j < y_max:           # Если разлет уменьшился, то сохраняем результат и записываем историю 
        
        y_max = y_j
        
        out_best = out
        print('\n', '** Удалось улучшить функцию!')
        
        # good_step = (good_step + steps * random_coef_vector) / 2
        
        beam_story.append(y_max)
        args_story.append(configuration.state_arguments_vector.copy())

    else:           # Если результат ухудшился, просто производим откат. 
        
        print('\n', '** Шаг не привел к улучшению...')

            
        
        
        


# 1. Сохраняем (Сериализация)
with open('out_simulation_1.pkl', 'wb') as file:
    pickle.dump(out, file)



print('beam_spread', beam_spread(out.y2))


    # Визуализируем результаты 
        
with open('out_simulation_1.pkl', 'rb') as file:
    out = pickle.load(file)

visual = Colider_simulator_visualization(configuration)

visual.trajectories = out_best.y2
visual.show_all()


print('\n')

print('Резюме', '\n')
print('beam_story', beam_story)
print('\n')
print('configuration.state_arguments_vector', configuration.state_arguments_vector)

print('args_story', args_story)

