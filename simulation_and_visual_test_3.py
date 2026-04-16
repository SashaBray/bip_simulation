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

coordinates = create_many_pionts(3, 5, 0.008, 0)
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
       




delta_array = np.ones_like(configuration.state_arguments_vector)

lerning_rate = 0.01

# steps = np.ones_like(configuration.state_arguments_vector) * lerning_rate

steps = np.array([1, -1]) * lerning_rate

n = 30

args_story = []
delta_array_i = []
beam_story = []



# Первый пробный запуск 
  
out = reality.simulate(speeds, 
                coordinates,
                charges,
                masses_rest,
                time_sim,
                #  delta_time=
                )

y_max = beam_spread(out.y2)     # Измеряем разлет частиц
beam_story.append(y_max)
args_story.append(configuration.state_arguments_vector.copy())

out_best = out


for i in range(n):

    
    random_coef_vector = np.random.uniform(-1, 1, size=1) # Производим случайное изменение одного из аргументов (ток одного из магнитов)
    value = random.uniform(-1, 1)
    
    # configuration.state_arguments_vector += good_step + steps * value
    configuration.state_arguments_vector += steps * value
    

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
        delta_array_i.append(y_j - y_max)
        
        beam_story.append(y_max)
        args_story.append(configuration.state_arguments_vector.copy())

    else:           # Если результат ухудшился, просто производим откат. 
        
        print('\n', '** Шаг не привел к улучшению...')
        configuration.state_arguments_vector -= steps * random_coef_vector
            
        
        
        


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
print('delta_array_i', delta_array_i)
print('\n')
print('configuration.state_arguments_vector', configuration.state_arguments_vector)

print('args_story', args_story)

