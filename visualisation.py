# import pyvista as pv

# python -m pip install pyvista 
# python -m pip install pyvista cascadio

# 1. Создаем или загружаем модельки (например, сферу и куб)
# model1 = pv.Sphere()
# model2 = pv.Cube()

# # 2. Задаем ориентацию (вращение)
# # Повернем куб на 45 градусов вокруг оси Z
# model2.rotate_z(45, inplace=True)

# # 3. Задаем расстояние (перемещение)
# # Сдвинем куб на 5 единиц по оси X
# model2.translate((5, 0, 0), inplace=True)

# # 4. Визуализация
# plotter = pv.Plotter()
# plotter.add_mesh(model1, color="red", show_edges=True)
# plotter.add_mesh(model2, color="blue", show_edges=True)

# plotter.add_axes() # Добавить оси координат для наглядности
# plotter.show()


import cascadio
import pickle
import os

import pyvista as pv
from colider_simulation_instruments_lib import *


way_direction = "C:\\Sasha\\Work\\3D models\\for_plotting\\"

stl_file_1 = "Assembly_6_pole_exp.STL"  # Имя вашей сборки
stl_file_2 = "Assembly_4_pole_exp.STL"
stl_file_3 = "rotoring_magnet_v2.STL"


# Загрузка модели
model_1 = pv.read(way_direction + stl_file_1)  # или .obj, .vtk
model_2 = pv.read(way_direction + stl_file_2)
model_3 = pv.read(way_direction + stl_file_3)


# mesh.rotate_x(20, inplace=True)

configuration_directory_link = 'C:\\Users\\Sasha Bray\\Desktop\\project\\database\\'
configuration_id = 3

configuration = Collider_configuration(configuration_directory_link, 
                                       configuration_id) # Класс, содержащий информацию о конфигурации установки

# all_fields = Field_generation_device() # Класс, соджержащий аппроксимации всех магнитных и электрических устройств. 



# # 2. Задаем ориентацию (вращение)
# # Повернем куб на 45 градусов вокруг оси Z
# model2.rotate_z(45, inplace=True)


# 2. Загружаем (Десериализация)
with open('out_simulation_0.pkl', 'rb') as file:
    out = pickle.load(file)

print('out', out.y1.shape)

traektory_1 = out.y2[:,0,:] * 1000


# # Допустим, это ваш массив nx3
# points = np.random.rand(100, 3).cumsum(axis=0) 

# 1. Создаем объект PolyData из точек
path = pv.PolyData(traektory_1)

# 2. Создаем индексы связей (lines)
# Формат: [количество_точек, id1, id2, ..., idN]
n_points = traektory_1.shape[0]
cells = np.full((n_points - 1, 3), 2, dtype=np.int_)
cells[:, 1] = np.arange(0, n_points - 1)
cells[:, 2] = np.arange(1, n_points)

path.lines = cells

# pl.add_mesh(path, color="blue", line_width=3, label="Траектория")

n_traectory = 2

model_1_coords = np.array([0, 0, 0]) * 1000
model_2_coords = np.array([5, 0, 0]) * 1000
model_3_coords = np.array([10, 0, 0]) * 1000

# model_3.translate((shift_3 + (-160), 200, 0), inplace=True)

model_3_1 = model_3.copy()
model_3_2 = model_3.copy()
model_3_3 = model_3.copy()

model_3_1.translate(model_1_coords, inplace=True)
model_3_2.translate(model_2_coords, inplace=True)
model_3_3.translate(model_3_coords, inplace=True)


# 3. Визуализация
pl = pv.Plotter()

pl.enable_parallel_projection()

opacity_comon = 0.2
color_1 = 'cyan' # "silver" 

pl.add_mesh(model_3_1, color=color_1, opacity=opacity_comon, show_edges=True, smooth_shading=True)
pl.add_mesh(model_3_2, color=color_1, opacity=opacity_comon, show_edges=True, smooth_shading=True)
pl.add_mesh(model_3_3, color=color_1, opacity=opacity_comon, show_edges=True, smooth_shading=True)

pl.add_mesh(path, color="blue", line_width=3, label="Траектория")

pv.global_theme.font.size = 6
pv.global_theme.font.title_size = 6


# pl.add_axes_at_origin(xlabel='X', ylabel='Y', zlabel='Z', line_width=2)


# pl.add_mesh(mesh, color="silver")

# 2. Указываем точку, к которой привязан текст (например, верхушка модели)
label_pos = np.array([0, 1, 1]) * 1000

# pl.add_point_labels([label_pos], ["Поворотный магнит"], 
#                     point_size=10,       # Размер точки-иконки
#                     point_color="red",   # Цвет точки
#                     font_size=20, 
#                     shape_color="white", # Цвет фона под текстом
#                     shape_opacity=0.8,   # Прозрачность фона
#                     show_points=True)    # Рисовать ли точку-иконку


pl.add_axes() # Добавляет цветные стрелки XYZ в углу 
# pl.add_legend() 
pl.show_grid() 
pl.show() 


# shift_1 = 0
# shift_2 = 2000
# shift_3 = 5000

# 3. Задаем расстояние (перемещение)
# Сдвинем куб на 5 единиц по оси X
# model_1.translate((shift_1 + 0, 0, 0), inplace=True)
# model_2.translate((shift_2 + 0, 100, 0), inplace=True)
# model_3.translate((shift_3 + (-160), 200, 0), inplace=True)


# plotter = pv.Plotter()
# plotter.add_mesh(model_1, color="red", show_edges=True)
# plotter.add_mesh(model_2, color="blue", show_edges=True)
# plotter.add_mesh(model_3, color="green", show_edges=True)


# axes = pv.CubeAxesActor(camera=plotter.camera)
# axes.bounds = model_1.bounds
# plotter.add_actor(axes)
# plotter.background_color = pv.Color('paraview', 0.6)

# plotter.add_axes() # Добавить оси координат для наглядности
# plotter.show()

