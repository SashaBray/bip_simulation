import time

import unittest
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


from colider_simulation_instruments_lib import *


configuration_directory_link = 'C:\\Users\\Sasha Bray\\Desktop\\project\\database\\'
configuration_id = 3

configuration = Collider_configuration(configuration_directory_link, 
                                       configuration_id) # Класс, содержащий информацию о конфигурации установки

all_fields = Field_generation_device() # Класс, соджержащий аппроксимации всех магнитных и электрических устройств. 

all_fields.add_configuration(configuration) # Передавем классу с апроксимациями класс конфигурации. Запускается процедура инициализации оберток полей.

x = np.array([0, 0, 0])


start_time = time.perf_counter()    # Засекаем время начала работы программы


# for i in range(1000):
#     B, E = all_fields.calculate(x)
#     """Эта функция пока не производит обратного преобразования магнитного поля в глобальные координаты. Вектора полей нужно помножать на 
# транспонированные мнк. + 
# Эта функция пока не умеет работать с симметрией, если область аппроксимации не граничит с одной из базовых плоскостей"""



coords_init = np.array([0, 0.05, 0.05])
coords_final = np.array([11, 0.05, 0.05])
steps = 10000
 
fields = []
 
coords_step = (coords_final - coords_init)/steps

for i in range(steps+1):
    
    coords_i = coords_step * i + coords_init 
    field_b, field_e = all_fields.calculate(coords_i)
    
    fields.append(field_b)


fields = np.array(fields)


end_time = time.perf_counter()
print(f"Затрачено времени: {end_time - start_time:.6f} сек.")

plt.scatter(np.arange(fields.shape[0]), fields[:,0])
plt.scatter(np.arange(fields.shape[0]), fields[:,1])
plt.scatter(np.arange(fields.shape[0]), fields[:,2])
plt.grid(True)
plt.show()
    
    

# print('B, E', B, E)

# coordinates = np.array([[0, 0, 0],               # Координаты трех частиц
#                [1e-6, 1e-6, 1e-6],
#                [1e-6, 1e-6, 1e-6]])

# С = 299792458
# persent = 0.999

# speeds = np.array([[С*persent, 0, 0],
#                    [С*persent, 0, 0],
#                    [С*persent, 0, 0]])

# Q = 1.60217653e-19
# M_e = 9.109383713928e-31

# charges = np.array([Q, Q, Q])
# masses = np.array([M_e, M_e, M_e])

# sim_results = Colider_simulator(all_fields)









