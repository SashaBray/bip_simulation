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

# x = np.array([0, 0, 0])

x = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]
              ])


start_time = time.perf_counter()    # Засекаем время начала работы программы



B, E = all_fields.calculate(x)
"""Эта функция пока не производит обратного преобразования магнитного поля в глобальные координаты. Вектора полей нужно помножать на 
транспонированные мнк. + 
Эта функция пока не умеет работать с симметрией, если область аппроксимации не граничит с одной из базовых плоскостей"""


print('\n', 'B', B)
print('\n', 'E', E)

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



# sim_results = Colider_simulator(all_fields, 
#                                 )









