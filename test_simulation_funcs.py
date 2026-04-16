
import time
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from colider_simulation_instruments_lib import *

reality = Colider_simulator()

coordinates = np.array([[-4, 0, 0],               # Координаты трех частиц
               [-4, 1e-6, 1e-6],
               [-4, 1e-6, 1e-6]])

С = 299792458       # Скорость света 
persent = 0.999

speeds = np.array([[С*persent, 0, 0],
                   [С*persent, 0, 0],
                   [С*persent, 0, 0]])

Q = -1.60217653e-19
M_e = 9.109383713928e-31

masses_rest = M_e * np.ones((3,1), dtype='float64') 
charges = Q * np.ones((3,1), dtype='float64') 

b_field = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

e_field = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]])

masses = np.array([M_e, M_e, M_e])

print('\n', 'masses_rest \n', masses_rest)

print('\n','relativistic_mass_electron \n', reality.relativistic_mass_electron(speeds, masses_rest))

print('\n','lorentz_force \n', reality.lorentz_force(charges, speeds, b_field) )
        
print('\n','coulomb_force \n', reality.coulomb_force(charges, e_field) )
        

lorents_forse_vector = reality.lorentz_force(charges, speeds, b_field)
masses_relatevistic = reality.relativistic_mass_electron(speeds, masses_rest)

print('\n','acceleration_vector \n', reality.acceleration_vector(lorents_forse_vector, masses_relatevistic))
        
        

configuration_directory_link = 'C:\\Users\\Sasha Bray\\Desktop\\project\\database\\'
configuration_id = 3

configuration = Collider_configuration(configuration_directory_link, 
                                       configuration_id) # Класс, содержащий информацию о конфигурации установки

all_fields = Field_generation_device() # Класс, соджержащий аппроксимации всех магнитных и электрических устройств. 
all_fields.add_configuration(configuration) # Передавем классу с апроксимациями класс конфигурации. Запускается процедура инициализации оберток полей.

        

reality.fields_func = all_fields

time_sim = 9/С
        
start_time = time.time()  # Фиксируем время начала

        
out = reality.simulate(speeds, 
                 coordinates,
                 charges,
                 masses_rest,
                 time_sim)

print('\n out.y2.shape', out.y2.shape, '\n')
print('\n out.y2', out.y2, '\n')

# plt.plot(out.y2[:,0], out.y2[:,1])
# plt.grid()
# plt.title('Line Graph')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# # print('Отклонение составило ',57.2958 * np.arctan(out.y1[-1][1]/out.y1[-1][0]), "градусов")

x = out.y2[:,0,0]
y = out.y2[:,0,1]
z = out.y2[:,0,2]

print('\n x', x, '\n')
print('\n y', y, '\n')
print('\n z', z, '\n')

# 2. Создаем 3D фигуру
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3. Рисуем линию траектории
ax.plot(x, y, z, label='Траектория', color='blue')

# Настройка подписей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_aspect('equal')

plt.show()

end_time = time.time()    # Фиксируем время конца
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time} секунд")


# 1. Сохраняем (Сериализация)
with open('out_simulation_0.pkl', 'wb') as file:
    pickle.dump(out, file)

# # 2. Загружаем (Десериализация)
# with open('hero_data.pkl', 'rb') as file:
#     loaded_hero = pickle.load(file)


