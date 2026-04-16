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




    # Инициализация виртуальной реальности и установки 
    
configuration_directory_link = 'C:\\Users\\Sasha Bray\\Desktop\\project\\database\\'
configuration_id = 8

configuration = Collider_configuration(configuration_directory_link, 
                                       configuration_id) # Класс, содержащий информацию о конфигурации установки

# reality = Colider_simulator()
all_fields = Field_generation_device() # Класс, соджержащий аппроксимации всех магнитных и электрических устройств. 
all_fields.add_configuration(configuration) # Передавем классу с апроксимациями класс конфигурации. Запускается процедура инициализации оберток полей.


    
        
start_time = time.time()  # Фиксируем время начала

coord_i = np.array([0, 0, 0])
fields = []

coords_init = np.array([1, 0.0, 0.0])
coords_final = np.array([4, 0.0, 0.0])
steps = 1000
 
 
coords_step = (coords_final - coords_init)/steps

for i in range(steps+1):
    
    coords_i = coords_step * i + coords_init 
    field_b, field_e = all_fields.calculate(coords_i)
          
    fields.append(field_b)


fields = np.array(fields)


end_time = time.time()    # Фиксируем время конца
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time} секунд")


visual = Colider_simulator_visualization(configuration)

# visual.trajectories = out.y2
# visual.show_all()

print('\n', 'fields', fields )
print('\n', 'fields.shape', fields.shape )


plt.scatter(np.arange(fields.shape[0]), fields[:,0])
plt.scatter(np.arange(fields.shape[0]), fields[:,1])
plt.scatter(np.arange(fields.shape[0]), fields[:,2])
plt.grid(True)
plt.show()
