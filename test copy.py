
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from colider_simulation_instruments_lib import *


"""Задаем путь и имя файла с информацией праметров магнитного устройства"""
path_to_file = r"C:\\Users\\Sasha Bray\\Desktop\\project\\database\\"      # Путь к файлу с данными о поле 
file_name = 'preprocessing_information.xlsx'                               # Название файла 

df_agregats_types = pd.read_excel(path_to_file + file_name)                           # Открываем файл
print('\n', df_agregats_types.columns)

"""Инициализируем экземпляр класса поля"""
six_pole_linse = Field_approximator(all_information = df_agregats_types.iloc[6])

"""Зпускаем процедуру чтения исходных файлов ansys и преобразование их в файлы кастомного формата"""
# six_pole_linse.prepare_an_approximation()
"""Запускаем процедуру чтения файлов, подготовленных предыдущей процедурой"""
six_pole_linse.read_field()


"""Зададим значение тока и координат"""
amper = 97

start_time = time.perf_counter()    # Засекаем время начала работы программы


coord_i = np.array([0, 0, 0])
fields = []


coords_init = np.array([-2, 0.0, 0.0])
coords_final = np.array([2, 0.0, 0.0])
steps = 1000
 
 
coords_step = (coords_final - coords_init)/steps

for i in range(steps+1):
    
    coords_i = coords_step * i + coords_init 
    field = six_pole_linse.calculate(coords_i, amper)
    
    # print('\n', 'field_i', field) 
            
    fields.append(field)


fields = np.array(fields)


end_time = time.perf_counter()
print(f"Затрачено времени: {end_time - start_time:.6f} сек.")

# print(fields.shape)

plt.scatter(np.arange(fields.shape[0]), fields[:,0])
plt.scatter(np.arange(fields.shape[0]), fields[:,1])
plt.scatter(np.arange(fields.shape[0]), fields[:,2])
plt.grid(True)
plt.show()


