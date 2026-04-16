
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from colider_simulation_instruments_lib import Field_approximator


"""Задаем путь и имя файла с информацией праметров магнитного устройства"""
path_to_file = r"C:\\Users\\Sasha Bray\\Desktop\\project\\database\\"      # Путь к файлу с данными о поле 
file_name = 'preprocessing_information.xlsx'                               # Название файла 

df_agregats_types = pd.read_excel(path_to_file + file_name)                           # Открываем файл
print('\n', df_agregats_types.columns)

"""Инициализируем экземпляр класса поля"""
six_pole_linse = Field_approximator(all_information = df_agregats_types.iloc[0])

"""Зпускаем процедуру чтения исходных файлов ansys и преобразование их в файлы кастомного формата"""
# six_pole_linse.prepare_an_approximation()
"""Запускаем процедуру чтения файлов, подготовленных предыдущей процедурой"""
six_pole_linse.read_field()


"""Зададим значение тока и координат"""
amper = 180

# x_init = np.array([-1, 0.00, 0.00])
# x_final = np.array([1, 0.00, 0.00])

# steps = 200

# x_step = (x_init - x_final)/steps

# fields = []
# coords_i = []


start_time = time.perf_counter()    # Засекаем время начала работы программы

# for i in range(steps+1):
#     x_coords = x_init + x_step
#     """Расчитываем поле в заданных координатах при заданном токе"""
#     B_field = six_pole_linse.calculate(x_coords, amper)     
    
#     fields.append(B_field)
#     coords_i.append(x_coords)

w = 0.02

x = np.linspace(-1, 1, int(2*1/w))
y = np.linspace(-1.e-01, 1.e-01, int(2*1.e-01/w))
z = np.linspace(-1.e-01, 1.e-01, int(2*1.e-01/w))

X, Y, Z = np.meshgrid(x, y, z)

coord_i = np.array([0, 0, 0])
fields = []

# for x_i in x:
#     print('hehe', x_i)

for x_i in x:
    for y_i in y:
        for z_i in z:
            coord_i = np.array([x_i, y_i, z_i])
            
            # print('coord_i', coord_i)

            field = six_pole_linse.calculate(coord_i, amper)
            
            fields.append(field)


fields = np.array(fields)

end_time = time.perf_counter() # Засекаем время окончания работы программы

fields_for_visual = fields[:,1]

print('fields_for_visual.shape', fields_for_visual.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Визуализация через scatter: c=V.flatten() задает цвет точек по значениям V
img = ax.scatter(X, Y, Z, c=fields_for_visual.flatten(), cmap='magma', alpha=0.2)
fig.colorbar(img) # Шкала значений
plt.show()


# fields = np.array(fields)
# coords_i = np.array(coords_i)

# execution_time = end_time - start_time # Вычисляем разницу
# print('\n', f"Время выполнения: {execution_time:.6f} секунд")
# print('\n', 'x_step', x_step)










