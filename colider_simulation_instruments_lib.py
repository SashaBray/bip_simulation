
import os
import re
import math

import numpy as np
import pandas as pd
import bdsim
import pyvista as pv

from numpy.linalg import norm
from scipy.spatial.distance import pdist



class InvalidDataError(Exception):
    """Исключение для некорректных данных."""
    pass


class Field_approximator(): 
    """Класс с набором инструментов для работы магнитными полями. Магнитное поле рассчитывается
    в программе ansys. Для разных токов формируются разные отчеты. Отчет представляет собой текстовый
    файл, где в начале указываются диапазоны сетки в глобальных координатах и шаг сетки. Ниже идут координаты и три проекции векторов поля. 
    
    Программа должна считывать исходные файлы ansys и преобразовывать их в пару файлов формата numpy. 
    
    Программа проверяет целостность подготовленных файлов. Обновляет их по требованию. """

    def __init__(self, all_information: np.iloc):
        """_summary_
        Args:
            init_files_way (str, optional): _description_. Defaults to "".
            afterpross_files_way (str, optional): _description_. Defaults to "".
        """
        self.all_information = all_information
        self.init_files_way = all_information['path_to_the_ANSYS_output_file']
        self.afterpross_files_way = all_information['path_to_the_after_processing_file']
        self.unit_si_dict = {'mm': 0.001}
        self.enable_messages = False
        
        # print('all_information', all_information)
        
        if str(all_information['B_or_E_type']) == "B":
            # print('*-*')
            self.field_type = True
        else:
            # print('*o*')
            self.field_type = False
        pass
    
    def ready_chek(self, restart=True):
        pass
    
    def unit_normalize(self, value: float, unit: str) -> float:
        
        value = float(value) * self.unit_si_dict[str(unit)]
        
        return value
    
    def open_ansys_grid_file(self, for_open: str) -> np.array:
        
        with open(for_open, 'r', encoding='utf-8') as file:
            content = file.read()
            
        lines = content.splitlines()
        splited_first = lines[0].split(']')
        coordinates_info = []
        coordinates_info_last = []
        
        for element in splited_first:
            coordinates_info.append(element.split("[")[-1])
        coordinates_info.pop(-1)
            
        # print('\n', coordinates_info, '\n')
        
        new_coordinates_info = [item + " " for item in coordinates_info]
        long_str = "".join(new_coordinates_info).split(" ")

        pattern = r"(-?\d+)([a-z]+)"

        for s in long_str:
            match = re.search(pattern, s)
            if match:
                value = match.group(1)  # Первая группа: число со знаком
                unit = match.group(2)   # Вторая группа: текст
                value = self.unit_normalize(value, unit) 
                coordinates_info_last.append(value)
        
        np_matrix = np.array(lines[2:])
        
        matrix = np.array([row.split() for row in np_matrix], dtype=np.float32)
        coord_array = np.array(coordinates_info_last)
        
        self.coordinates_info_last = coordinates_info_last
        
        if self.enable_messages:
            print('\n', 'coordinates_info_last', coordinates_info_last, '\n') 
            # print('\n', 'matrix', matrix) 
            # print('matrix_shape', matrix.shape, '\n') 
        
        return coord_array, matrix
    
    def coords_by_index(self, x_max: float, x_min: float, i_max: int, i: int) -> float:
        x_i = (x_max - x_min) * i / i_max + x_min
        return x_i
    
    def index_by_coords(self, x_max: float, x_min: float, i_max: int, x_i: float) -> int:
        i = (x_i - x_min) * i_max / (x_max - x_min) 
        # print("i", i)
        return int(i)
    
    def make_field_tensor(self, matrix: np.array, coords: np.array) -> np.array:
        
        x_lable_size = int((coords[3] - coords[0])/coords[6])
        y_lable_size = int((coords[4] - coords[1])/coords[7])
        z_lable_size = int((coords[5] - coords[2])/coords[8])
        
        # print('\n', 'x_lable_size',  x_lable_size, 'y_lable_size', y_lable_size, 'z_lable_size', z_lable_size, '\n')
        
        shape = (x_lable_size+1, y_lable_size+1, z_lable_size+1, 3) # Глубина, Строки, Столбцы
        tensor = np.empty(shape)
        
        # print('arr', tensor)
        # print('arr', matrix.shape)
        
        x_min, y_min, z_min = coords[0], coords[1], coords[2]
        x_max, y_max, z_max = coords[3], coords[4], coords[5]
        x_step, y_step, z_step = coords[6], coords[7], coords[8]
        
        for row in matrix:
            x_i, y_i, z_i = row[0], row[1], row[2]
            Bx, By, Bz = row[3], row[4], row[5]
            
            field_vector = [Bx, By, Bz]
            
            i_x = self.index_by_coords(x_max, x_min, x_lable_size, x_i)
            i_y = self.index_by_coords(y_max, y_min, y_lable_size, y_i)
            i_z = self.index_by_coords(z_max, z_min, z_lable_size, z_i)
            
            # print('tensor[i_x][i_y][i_z]', tensor[i_x][i_y][i_z])
            tensor[i_x][i_y][i_z] = field_vector
        
        return tensor
    
    def nawes_of_ready_files(self):
        
        arguments = self.all_information['device_model_name'] + '_arguments' + '.csv'
        shapes = self.all_information['device_model_name'] + '_shapes' + '.csv'
        field = self.all_information['device_model_name'] + '_field' + '.csv'
        coordinates = self.all_information['device_model_name'] + '_coordinates' + '.csv'
        path = self.all_information['path_to_the_after_processing_file'].replace('\\', os.sep) + '\\'.replace('\\', os.sep)
        
        return [path + arguments, path + shapes, path + field, path + coordinates, path]
    
    def save_field(self):
        
        names_for_open = self.nawes_of_ready_files()
        
        shapes = np.array(self.argument_field.shape)
        
        # print('self.argument_field', self.argument_field)
        
        np.savetxt(names_for_open[1], shapes, delimiter=',') 
        np.savetxt(names_for_open[0], self.argument_list, delimiter=',', fmt='%s') 
        np.savetxt(names_for_open[2], self.argument_field.reshape(shapes[0], -1), delimiter=',', fmt='%s') 
        np.savetxt(names_for_open[3], self.coordinates_info_last, delimiter=',', fmt='%s') 
        
        pass
    
    def prepare_an_approximation(self):
        files = os.listdir(self.init_files_way)
        if self.enable_messages:
            print(files)
        
        argument_list = []
        argument_field = []
        
        for file in files:
            """Происходит перебор всех файлов в папке, и если среди них есть имя с ожидаемым патерном
            этот файл обрабатывается и на его основе создается тензор поля. Тензор состоит из нескольких 
            трехмерных тензоров, соответствующих распределению поля при определенном токе. Чтобы узнать, какой 
            тензор какому току соответствует, существует массив argument_list длина которого совпадает с количеством 
            тензоров полей. Чтобы найти нужный ток, нужно проверить под каким индектом такой ток леджит в этом массиве, 
            и обратиться к тензору под таким же номером."""
            
            for_open = self.init_files_way.replace('\\', os.sep) + '\\'.replace('\\', os.sep) + file
            
            match = re.search(r'A(\d+)', file)
            if match:
                file_argument = match.group(1)
                coords, matrix = self.open_ansys_grid_file(for_open)
                
                sorted_tensor = self.make_field_tensor(matrix, coords)
                argument_list.append(file_argument)
                argument_field.append(sorted_tensor)
        
        argument_list = np.array(argument_list)
        argument_field = np.array(argument_field)  
        
        if self.enable_messages:
            print('\n', 'argument_list', argument_list)
            print('\n', 'argument_field.shape', argument_field.shape, '\n')
        
        shapes = np.array(argument_field.shape)
        
        self.argument_list = argument_list
        self.argument_field = argument_field
        
        self.save_field()

        pass
    
    def find_real_ranges(self):
        
        """Функция, инициализирующая дополнительные параметры поля, которые необходимо определить один раз 
        заранее, перед тем, как приступить к вычислениям"""
        
        if self.argument_list_sorted.shape == ():
            maximum_current = self.argument_list_sorted
        else:
            maximum_current = self.argument_list_sorted[-1]
        
        
        # Массив с коэффициентами симметрии. Три коэффициента пространства и один коэффициент симметрии поля. 
        symetric_array = [self.all_information['x_symmetry'], self.all_information['y_symmetry'], 
                                self.all_information['z_symmetry'], self.all_information['I_symmetry']]
        
        # Массив с границами допустимых значенениях параметров. Координаты и значения тока. Границы не учитывают симметрию. 
        coordinates_info_last_I = [self.coordinates_info_last[0], self.coordinates_info_last[1],
                                 self.coordinates_info_last[2], 0.0,
                                 self.coordinates_info_last[3], self.coordinates_info_last[4],
                                 self.coordinates_info_last[5], maximum_current]
        
        """В данном цикле создается массив, содержащий границы координат и тока с учетом заданной симметрии"""
        real_borders = []
        image_borders = []

        for i in range(4):
            range_i_min = coordinates_info_last_I[i]
            range_i_max = coordinates_info_last_I[i+4]
            simetry_i = symetric_array[i]
            
            image_borders.append(range_i_min)
            image_borders.append(range_i_max)

            if simetry_i == 0:
                real_borders.append(range_i_min)
                real_borders.append(range_i_max)
                
            elif simetry_i == 1 or simetry_i == -1:
                if range_i_min == 0:
                    real_borders.append(-range_i_max)
                    real_borders.append(range_i_max)
                    
                elif range_i_max == 0:
                    real_borders.append(range_i_min)
                    real_borders.append(-range_i_min)
                else: 
                    real_borders.append(range_i_min)
                    real_borders.append(range_i_max)
                    print('Указанные границы симметричного поля не связаны с основными плоскостями СК')
        
        self.real_borders = np.array(real_borders)
        self.symetric_array = np.array(symetric_array)
        self.coordinates_info_last_I = np.array(coordinates_info_last_I)
        self.image_borders = np.array(image_borders)
        
        
        """Массив с шагами сектри поля по трем направлениям"""
        step = np.array([self.image_borders[1] - self.image_borders[0], 
                    self.image_borders[3] - self.image_borders[2],
                    self.image_borders[5] - self.image_borders[4]
                    ])
        self.step = step/(np.array(self.field.shape[1:4]) - np.array([1, 1, 1]))
        
        """Вектор сдвига границ без учета симметрии относительно границ с учетом симметрии. 
        Нужен для корректного расчета промежуточных значений поля между узлами сетки аппроксимации"""
        self.shift_vector_real_image_borders =  np.array([image_borders[0], image_borders[2], image_borders[4]])


        if self.enable_messages:    # Переменная, разрешающая вывод сообщений, облегчающих отладку. 
            print('\n', 'self.real_borders', self.real_borders, '\n')
            print('\n', 'self.symetric_array', self.symetric_array, '\n')
            print('\n', 'self.coordinates_info_last_I', self.coordinates_info_last_I, '\n')
            print('\n', 'image_borders', self.image_borders)
        
        pass
    
    
    def read_field(self):
        """
        1. Процедура считывает заранее подготовленные файлы. Файлы имеют стандартные названия. 
        2. Запускает процедуру рассчтеа параметров сетки, которые потребуются при рассчете поля. 
        """
        names_for_open = self.nawes_of_ready_files()
        
        self.shapes = np.genfromtxt(names_for_open[1], delimiter=',') 
        self.argument_list = np.genfromtxt(names_for_open[0], delimiter=',') 
        self.coordinates_info_last = np.genfromtxt(names_for_open[3], delimiter=',') 
        self.field = np.genfromtxt(names_for_open[2], delimiter=',') 
        
        self.argument_list = np.array(self.argument_list)
        
        self.field = self.field.reshape(self.shapes.astype(int))
        self.argument_list_sorted = self.argument_list.copy()
        
        print('\n', 'self.argument_list_sorted', self.argument_list_sorted.shape)
        
        if self.argument_list_sorted.shape == ():
            self.argument_list_sorted = self.argument_list_sorted
        else:
            self.argument_list_sorted = np.sort(self.argument_list_sorted)
        
        
        self.find_real_ranges()
        
        pass
    
    def check_borders(self, x_coordinates: np.array, argument: float, borders) -> list:
        in_borders_bool_list = []        
        for i in range(3):
            coordinate_i = x_coordinates[i]
            if coordinate_i > borders[i*2] and coordinate_i < borders[i*2+1]:
                """Если координата лежит в мнимых границах... То есть границах действия поля, с учетом симметрии
                Значит функция в целом способна рассчитать поле с такими координатами"""
                in_borders_bool_list.append(True)
            else:
                in_borders_bool_list.append(False)
                
        if argument > borders[-2] and argument < borders[-1]:
            in_borders_bool_arg = True
        else:
            in_borders_bool_arg = False
            
        return np.array(in_borders_bool_list), in_borders_bool_arg
    

    def find_real_coords(self, x_coordinates: np.array, argument: float):
        """Рассчитаем практические координаты в тензоре, коэффициенты для соблюдения симметрии и 
        практический аргумент тока. И сохраним результаты в self"""
        
        in_imabe_borders, in_arg_bord = self.check_borders(x_coordinates, argument, self.image_borders)
        
        if self.enable_messages:
            print('\n', 'in_imabe_borders' , in_imabe_borders)
        
        real_x_coordinates = x_coordinates.copy()
        real_argument = argument
        sumetric_coeffs = []
        
        for i in range(3):
            range_i = in_imabe_borders[i]
            if range_i:
                sumetric_coeffs.append(1)
            else:
                sumetric_coeffs.append(self.symetric_array[i])
                real_x_coordinates[i] = -real_x_coordinates[i]
                
        range_i = in_arg_bord
        if range_i:
            sumetric_coeffs.append(1)
        else:
            sumetric_coeffs.append(self.symetric_array[-1])
            real_argument = -real_argument 
        
        self.real_argument = real_argument
        self.real_x_coordinates = real_x_coordinates
        self.sumetric_coeffs = sumetric_coeffs

        pass
    
    # def find_real_args(self, argument: float):
    #     """Если аргумент выходит за пределеы дискретно заданной сетки (току больше, чем было измерено во 
    #     время численного моделирования), то поле будет рассчитываться на основе самого большого известного поля"""
    #     max_arg = self.argument_list_sorted[-1]
    #     if max_arg <= argument:
    #          argument = max_arg
        
    #     return argument
    
    def find_intermediate_vector(self, vector_1, vector_2, step, local_argument):
        """Функция получает два вектоа, шаг между ними и промежуточное значение 
        а возвращает промежуточный вектор между двумя данными. 
        Подразумевается, что первый вектор расположен левее на оси координат. А локальный 
        аргумент - расстояние между левым вектором и искомым."""
        
        vector_new = ((vector_2 - vector_1)/step) * local_argument + vector_1
        
        return vector_new
    
    def find_two_closest_tensors(self, argument):
        """Определяем два длижайших по аргументу тензоров (наиболее близкие токи)"""

        if self.argument_list_sorted.shape == ():
            index_min, index_max = 0, 0
            
        else:
            # Находит индекс, куда нужно вставить v, чтобы сохранить порядок
            index = np.searchsorted(self.argument_list_sorted, argument, side='right')
            if index == self.argument_list_sorted.shape[0]:
                agr_min = self.argument_list_sorted[index - 1]
                agr_max = self.argument_list_sorted[index - 1]
            else:
                agr_min = self.argument_list_sorted[index - 1]
                agr_max = self.argument_list_sorted[index]
            
            index_min = np.where(self.argument_list == agr_min)[0][0]
            index_max = np.where(self.argument_list == agr_max)[0][0]
            
            self.step_arg = agr_max - agr_min
            self.local_arg_amper = argument - agr_min
            
        if self.enable_messages:
            print('\n', 'self.argument_list', self.argument_list, '\n')
            print('\n', 'self.argument_list_sorted', self.argument_list_sorted, '\n')
            print('\n', 'argument', argument, '\n')
            print('\n', 'index', index, '\n')
            print('\n', 'agr_min/agr_max', agr_min, agr_max, '\n')
            print('\n', 'index_min/index_max', index_min, index_max, '\n')
        
        return index_min, index_max
    
    def get_two_closest_tensors(self, argument, 
                                argument_index_min, 
                                argument_index_big):
        
        tensor_1, tensor_2 = self.field[argument_index_min], self.field[argument_index_big]
    
        return tensor_1, tensor_2
    
    def find_vector_by_coords_in_tensor(self, argument, argument_index):
        """Страшная функция по поочередному билинейному сокращению векторов...."""
        
        """Найдем индексы для восьми векторов, которые будут подвергнуты линейной интерполяции"""
        x_1 = self.index_by_coords(self.image_borders[1], 
                                    self.image_borders[0],
                                    self.shapes[1]-1, 
                                    self.real_x_coordinates[0])
        
        y_1 = self.index_by_coords(self.image_borders[3], 
                                   self.image_borders[2],
                                    self.shapes[2]-1, 
                                    self.real_x_coordinates[1])
        
        z_1 = self.index_by_coords(self.image_borders[5], 
                                    self.image_borders[4],
                                    self.shapes[3]-1, 
                                    self.real_x_coordinates[2])
        
        x_2 = x_1 + 1
        y_2 = y_1 + 1
        z_2 = z_1 + 1
        
        self.less_coords_vector = np.array([self.coords_by_index(self.image_borders[1], self.image_borders[0], self.shapes[1]-1, x_1),
                                            self.coords_by_index(self.image_borders[3], self.image_borders[2], self.shapes[2]-1, y_1),
                                            self.coords_by_index(self.image_borders[5], self.image_borders[4], self.shapes[3]-1, z_1),
                                            ])
 
        self.local_arg = self.real_x_coordinates - self.less_coords_vector
        
        if self.enable_messages:
            print('\n', 'x_1, y_1, z_1', x_1, y_1, z_1)
        
        tensor = self.field[argument_index]
        
        a = tensor[x_1][y_1][z_2]
        b = tensor[x_1][y_2][z_2]
        c = tensor[x_1][y_1][z_1]
        d = tensor[x_1][y_2][z_1]
        
        aa = tensor[x_2][y_1][z_2]
        bb = tensor[x_2][y_2][z_2]
        cc = tensor[x_2][y_1][z_1]
        dd = tensor[x_2][y_2][z_1]
        
        ab = self.find_intermediate_vector(a, b, self.step[1], self.local_arg[1])
        cd = self.find_intermediate_vector(c, d, self.step[1], self.local_arg[1])
        
        aabb = self.find_intermediate_vector(aa, bb, self.step[1], self.local_arg[1])
        ccdd = self.find_intermediate_vector(cc, dd, self.step[1], self.local_arg[1])
        
        ab_cd = self.find_intermediate_vector(ab, cd, self.step[2], self.local_arg[2])
        aadd_ccdd = self.find_intermediate_vector(aabb, ccdd, self.step[2], self.local_arg[2])
        
        result_vector = self.find_intermediate_vector(ab_cd, aadd_ccdd, self.step[0], self.local_arg[0])
        
        
        return result_vector
        
        
    def start_approximation(self, argument, in_real_borders, in_arg_bord):
        
        if self.enable_messages:
            print('\n', 'self.step', self.step, '\n')
        
        argument_index_min, argument_index_big = self.find_two_closest_tensors(argument)
        
        if argument_index_min == argument_index_big:
            vector_secoen = self.find_vector_by_coords_in_tensor(argument, argument_index_min)
            
            vector_third = self.find_intermediate_vector(np.array([0, 0, 0]), vector_secoen, self.argument_list_sorted, argument)
                        
            
        else:
            vector_first = self.find_vector_by_coords_in_tensor(argument, argument_index_min)
            vector_secoen = self.find_vector_by_coords_in_tensor(argument, argument_index_big)
                        
            vector_third = self.find_intermediate_vector(vector_first, vector_secoen , self.step_arg, self.local_arg_amper)
                        
        # vector_third = vector_third * self.sumetric_coeffs[-1]
            
        vector_third = vector_third * self.sumetric_coeffs[:-1]
        self.vector_third = vector_third
        
        if self.enable_messages:
            print('\n', 'self.sumetric_coeffs', self.sumetric_coeffs, '\n')
            print('\n', 'vector_third', vector_third, '\n')
        
        return vector_third
    
    
    def calculate(self, x_coordinates: np.array, argument: float) -> tuple:
        """Функция, рассчитывающая магнитное поле при произвольном токе и при произвольной координате, 
        в рамках заданного диапазона. Если ток выйдет за пределы имеющегося диапазона, метод вернет предупреждение
        и вектор, рассчитанный на экстрополяции ближайших известных значений апроксимации.
        Если координаты выйдут за пределы установленного диапазона, функция вернет нулевой вектор."""
        
        
        in_real_borders, in_arg_bord = self.check_borders(x_coordinates, argument, self.real_borders) # Проверяем, возможно ли рассчитать поле для заявленных координат 
        check = in_real_borders.prod()
        self.x_coordinates = x_coordinates
        
         
        if not(check):
            """Если координата находится за пределами фактических границ поля, функция позвращает нулевой вектор"""
            if self.enable_messages:
                print('\n', 'Запрашиваемые координаты за пределами дискретно заданного поля', '\n')
            # print('\n', self.x_coordinates)
            vector_third = np.array([0.0, 0.0, 0.0])
        
        else:
            """В противном случае программа перерассчитывает координаты с учетом симметрии, сохраняет коэффициенты и
            определяет координаты восьми ближайших известных векторов, чтобы приступить к билинейной интерполяции"""
            if self.enable_messages:
                print('\n', 'Запрашиваемые координаты находятся в пределах дискретно заданного поля', '\n')
        
            self.find_real_coords(x_coordinates, argument)
            
            # argument = self.find_real_args(argument)
            vector_third = self.start_approximation(argument, in_real_borders, in_arg_bord)
            # return vector_third
        
        
        if self.field_type:
            return vector_third, np.array([0, 0, 0])    
        else:
            return np.array([0, 0, 0]), vector_third

            
        pass



class Field_generation_device(): 
    """A class of magnetic devices containing instances of non-intersecting grids of magnetic field approximations.
    A magnetic device has coordinates of its position and orientation angles.
    """

    def __init__(self, coordinates = np.array([0, 0, 0]),  # coordinates: np.array([x, y, z])
                 angels = np.array([0, 0, 0]),            # angels: np.array([крен, тангаж, рыскание]) (roll, pitch, yaw)
                 list_magnet_aproximations = [], 
                 list_magnet_fields_id = [],
                #  list_electro_approximtions = [],
                #  list_electro_fields_id = [],
                 angles_units = "degrees", #  "degrees"/"radians"
                 all_addition_inf = 0
                 ):
       
        
        self.coordinates = coordinates
        self.angels = angels
        self.angles_units = angles_units
        
        self.list_magnet_aproximations = list_magnet_aproximations
        self.list_magnet_fields_id = list_magnet_fields_id
        # self.list_electro_approximtions = list_electro_approximtions
        # self.list_electro_fields_id = list_electro_fields_id
        
        self.all_addition_inf = all_addition_inf
        
        self.direction_cosine_matrix = self.get_direction_cosine_matrix(angels, angles_units)
       
        pass
    
    def get_direction_cosine_matrix(self, angels: np.array, angles_units = "degrees") -> np.array:
        """The function accepts a vector with orientation angles and returns a matrix of direction cosines. 
        It's important to specify the units of measurement used for the orientation angles.

        Args:
            angels (np.array): Angles of successive rotations around the x, y and z axes. Euler angles. Roll, pitch, yaw. Right-hand coordinate system. 
            angles_units (str, optional): "degrees"/"radians". Defaults to "degrees".

        Raises:
            ValueError: The orientation angle array must have size 3. 
            
            TypeError: The coordinate array must contain numeric values.

        Returns:
            np.array: 3x3 direction cosine matrix
        """
        
        angels = np.asarray(angels) # Превращаем в массив, если это список
        if angels.shape != (3,):
            raise ValueError("Array must have exactly 3 elements")
        if not np.issubdtype(angels.dtype, np.number):
            raise TypeError("Array must contain numbers")
        
        if angles_units == "degrees":
            angle_radians = np.radians(angels)
             
        
        # Задание углов в радианах. [рыскание, тангаж, крен]
        gamma = -angle_radians[0] # крен 
        psi = angle_radians[1]  # тангаж 
        theta = -angle_radians[2] # рыскание

        # Матрица поворота по крену.
        C_gamma = np.array(
            [
                [1, 0, 0],
                [0, np.cos(gamma), np.sin(gamma)],
                [0, -np.sin(gamma), np.cos(gamma)],
            ]
        )

        # Матрица поворота по рысканию.
        C_psi = np.array(
            [
                [np.cos(psi), 0, -np.sin(psi)],
                [0, 1, 0],
                [np.sin(psi), 0, np.cos(psi)],
            ]
        )

        # Матрица поворота по тангажу.
        C_theta = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        # Матрица для последовательности psi->theta->gamma.
        result_matrix = C_gamma @ C_theta @ C_psi
        
        
        return result_matrix
    
    
    def find_local_coordinates(self, coordinates: np.array, shift_vector: np.array, rotation_matrix: np.array) -> np.array:
        """The function allows you to switch between coordinate systems.

        Args:
            coordinates (np.array): Initial coordinates
            shift_vector (np.array): The distance vector between the centers of two coordinate systems
            rotation_matrix (np.array): a direction cosine matrix storing the rotation of one coordinate system relative to another.

        Returns:
            np.array: vector in another coordinate system
        """
        
        
        coordinates = np.asarray(coordinates) # Превращаем в массив, если это список
        if coordinates.shape != (3,):
            raise ValueError("Array must have exactly 3 elements")
        if not np.issubdtype(coordinates.dtype, np.number):
            raise TypeError("Array must contain numbers")
        
        shift_vector = np.asarray(shift_vector) # Превращаем в массив, если это список
        if shift_vector.shape != (3,):
            raise ValueError("Array must have exactly 3 elements")
        if not np.issubdtype(shift_vector.dtype, np.number):
            raise TypeError("Array must contain numbers")
        
        rotation_matrix = np.asarray(rotation_matrix) # Превращаем в массив, если это список
        if rotation_matrix.shape != (3,3):
            raise ValueError("Array must have exactly 3 elements")
        if not np.issubdtype(rotation_matrix.dtype, np.number):
            raise TypeError("Array must contain numbers")
        
        # print('local_coordinates', local_coordinates)
                
        
        local_coordinates = rotation_matrix @ (coordinates - shift_vector)
        
        return local_coordinates
        
        
    def calculate_a_field_from_list(self, x_coordinates, list_aproximations, list_fields_id) -> np.array:
        """The function iterates over the lists and either gets a field or passes the calculations down to the next level."""

        vector_b_acc, vector_e_acc = np.array([0, 0, 0]), np.array([0, 0, 0])
        
        # print('\n', 'list_fields_id', list_fields_id)
        
        for i in range(len(list_aproximations)):
            fields_id = list_fields_id[i]
            object_instance = list_aproximations[i]
            
            # print('fields_id', fields_id)
            
            shift_vector, rotation_matrix = self.all_addition_inf.position(fields_id)
            rotation_matrix_t = self.all_addition_inf.mnk_t[fields_id]
            x_coordinates_local = self.find_local_coordinates(x_coordinates, shift_vector, rotation_matrix)
            
            if isinstance(object_instance, Field_generation_device):
                # argument = self.all_addition_inf.argument(fields_id)
                vector_b, vector_e =+ object_instance.calculate(x_coordinates_local)
                
                """После вычисления поля, нужно преобразовать это поле из локальных координат в глобальные, 
                то есть повернуть, согласно тому, как повернуто устройтсво относительно глобальной системы координат, 
                то есть умножаем полученные вектора на транспонированную мнк"""
                
                vector_b_acc, vector_e_acc = vector_b_acc + rotation_matrix_t @ vector_b, vector_e_acc + rotation_matrix_t @ vector_e
                
                pass
                
            elif isinstance(object_instance, Field_approximator):
                argument = self.all_addition_inf.argument(fields_id)
                vector_b, vector_e = object_instance.calculate(x_coordinates_local, argument)
                
                vector_b_acc, vector_e_acc = vector_b_acc + rotation_matrix_t @ vector_b, vector_e_acc + rotation_matrix_t @ vector_e
                
                pass
        
        
        return vector_b_acc, vector_e_acc
    
    
    def calculate(self, x_coordinates: np.array) -> tuple:
        
        if x_coordinates.ndim == 1:
            """If coordinates are obtained for one particle..."""
            
            B_field, E_field = self.calculate_a_field_from_list(x_coordinates, self.list_magnet_aproximations, self.list_magnet_fields_id)
            # E_field = self.calculate_a_field_from_list(x_coordinates, self.list_electro_aproximations, self.list_electro_fields_id)
            
            pass
        
        elif x_coordinates.ndim == 2:
            """If coordinates are obtained for several particles..."""
            
            B_field, E_field = np.zeros_like(x_coordinates, dtype='float64'), np.zeros_like(x_coordinates, dtype='float64')

            for i, coord_i in enumerate(x_coordinates):
                B_field_i, E_field_i = self.calculate_a_field_from_list(
                    coord_i,
                    self.list_magnet_aproximations,
                    self.list_magnet_fields_id
                )
                
                # print('\n','i', i)
                # print('B_field_i.dtype - ', B_field_i.dtype)
                # print('B_field[i]', B_field[i].shape)
                # print('B_field_i - ', B_field_i)
                # print('B_field_i.shape - ', B_field_i.shape)
                # print('type(B_field_i)', type(B_field_i))
                
                B_field[i] = B_field_i
                E_field[i] = E_field_i
                
                # print('B_field[i] - ', B_field[i])

            
        else:
            raise ValueError(f"Неподходящая размерность: {x_coordinates.ndim}. Ожидалось 1 или 2.")
    
        
        return B_field, E_field
    
        
    
    def add_configuration(self, configuration):
        """Инициализируем обертки для всех полей"""
        
        
        self.all_addition_inf = configuration
        self.unique_fields_id = configuration.df_agregats_config_id['field_id'].unique().tolist() # Уникальные магнитные устройства
        
        indexses = []   # Номера строк в таблице с полями, которые соответсвуют нашим уникальным устройствам. 
        unique_fields_id = []
        
        for i in range(len(self.unique_fields_id)):
            
            id_for_find = self.unique_fields_id[i]
            machin_index = configuration.df_agregats_types[configuration.df_agregats_types['field_id'] == id_for_find].index.tolist()
            
            # print("machin_index", machin_index[0])
            unique_fields_id.append(configuration.df_agregats_types.iloc[i]['field_id'])
            
            if len(machin_index) != 0:
                indexses.append(machin_index[0])
                
            else: 
                raise InvalidDataError("""The configuration file (machines_composition_detailed) specifies the device id, 
                                       which is not listed in the device database (preprocessing_information)""")
                
        unique_fields = []
        
        print('indexses', indexses)
        
        for id in indexses: # Создаем список уникальных полей 
            # print('*')
            field_obj = Field_approximator(all_information = configuration.df_agregats_types.iloc[id])
            field_obj.prepare_an_approximation()
            field_obj.read_field()
            unique_fields.append(field_obj)
            
        # print('unique_fields', unique_fields)
        # print('indexses', indexses)
        
        merget_df = pd.merge(configuration.df_agregats_types, 
                             configuration.df_agregats_config_id,
                             on='field_id')
        
        print('merget_df', '\n', merget_df)
        
        print('unique_fields_id', unique_fields_id)
        
        global_field_id = []
        global_fields_list = []
        
        
        for index, row in  configuration.df_agregats_config_id.iterrows():
            
            field_id_i = row['field_id']
            # print('\n', 'self.unique_fields_id', self.unique_fields_id)
            # print('field_id_i', field_id_i)
            
            field_id = self.unique_fields_id.index(field_id_i)
            
            global_fields_list.append(unique_fields[field_id])
            global_field_id.append(index)
            
            pass
        
        self.list_magnet_aproximations = global_fields_list
        self.list_magnet_fields_id = configuration.df_agregats_config_id['serial_number'].tolist()
        
    
        pass
    
    
class Collider_configuration(): 
    
    
    def __init__(self, configuration_directory_link='', 
                    configuration_id: str='',
                    file_fields = 'preprocessing_information.xlsx', 
                    file_configuration = 'machines_composition_detailed.xlsx',                  
                    ):
        
        # if df_agregats_types == 0 and df_agregats_config
        
        self.configuration_directory_link = configuration_directory_link
        self.configuration_id = int(configuration_id)
        self.file_fields = file_fields
        self.file_configuration = file_configuration
        
        if configuration_directory_link != '':
            # print('*')
            self.create_list_of_orientation_matrices_and_vectors()
        
        pass
    
    
    def create_list_of_orientation_matrices_and_vectors(self):
        """_summary_
        """
        
        self.df_agregats_types = pd.read_excel(self.configuration_directory_link + self.file_fields)
        df_agregats_config = pd.read_excel(self.configuration_directory_link + self.file_configuration)
        
        self.df_agregats_config_id = df_agregats_config[df_agregats_config['Cofiguration_id'] == self.configuration_id]    
        self.df_agregats_config_id['serial_number'] = range(len(self.df_agregats_config_id))
        
        self.df_agregats_types['agregat_number'] = range(len(self.df_agregats_types))
        
        
        print(self.df_agregats_config_id)
        
        
        magnet_device = Field_generation_device()
        
        mnk = []    # Матрица направляющих косинусов (ориентация магнитных устройств в глобальной системе координат)
        mnk_t = []  # Обратная матрица направляющих косинусов 
        shifts = [] # Координаты магнитных устройств в глобальной системе координат 
        state_arguments_vector = [] # Текущие значения токов или напряжения магнитных устройств 
        argument_coef = []  # Кооэфициенты при значениях аргументов (токов или напряжений)
        speed = []  # Максимальная скорость, с которой аргумент может меняться
        aim_argument_coef = []  # Целевое значение аргумента, если прямо во время моделирования происходит переходный процесс изменения значения аргумента
        
        
        for index, row in self.df_agregats_config_id.iterrows():
                        
            alpha, betta, gamma = row["alfa"], row["betta"], row["gamma"]
            shift_x, shift_y, shift_z = row["x"], row["y"], row["z"]
            argument_i = row["init_arg"]
            speed_i = row["rate_change_argument"]
            argument_coef_i = row["arg_coef"]
        
            degris = np.array([alpha, betta, gamma])
            matrix_i = magnet_device.get_direction_cosine_matrix(degris)
            
            shift_i = np.array([shift_x, shift_y, shift_z])
            
            mnk.append(matrix_i)
            mnk_t.append(np.linalg.inv(matrix_i))
            shifts.append(shift_i)
            state_arguments_vector.append(argument_i)
            argument_coef.append(argument_coef_i)
            speed.append(speed_i)
        
        self.mnk = np.array(mnk)
        self.mnk_t = np.array(mnk_t)
        self.shifts = np.array(shifts)
        
        self.state_arguments_vector = state_arguments_vector
        self.argument_coef = argument_coef
        self.speed = speed
        
        pass
    
    def position(self, fields_id: int) -> tuple:
        
        vector, matrix = self.shifts[fields_id], self.mnk[fields_id]
        
        return vector, matrix
        
    def argument(self, fields_id: int) -> float:
        
        result = self.state_arguments_vector[fields_id] * self.argument_coef[fields_id]
        
        return result
    
    
    
class Colider_simulator():
    
    def __init__(self):
        
        self.C= 299792458
        self.с_2 = self.C**2
        self.fields_func = None
        
        
        pass
    
    def relativistic_mass_electron(self, v: np.array, m: np.array) -> float:
        
        ones_array = np.ones_like(m)
        
        v_squared_norms = (v**2).sum(axis=1, keepdims=True)     # norm v**2
        
        coefs = ones_array - v_squared_norms/(ones_array * self.с_2)
        
        coefs_sqrt = np.sqrt(coefs)
        
        # try:
        #     coefs_sqrt = np.sqrt(coefs)
        # except:
        #     print('coefs', coefs)
        #     # coefs_sqrt 
        
        m_v = m * coefs_sqrt
        
        return m_v
    
    def lorentz_force(self, q, v, b_field) -> np.array:
        # F_l = qvb
        vect_vb = np.cross(v, b_field)
        lorentz_f = q * vect_vb
        return lorentz_f
    
    def coulomb_force(self, q, e_field) -> np.array:
        # F_c = E * q
        coulomb_f = e_field * q
        return coulomb_f
    
    def acceleration_vector(self, forse, masses) -> np.array:
        acceleration_v = forse / masses
        return acceleration_v
    
    def tensor_to_vector(self, matrix):
        vector = matrix.ravel()
        return vector
    
    def vector_to_tensor(self, vector):
        matrix = vector.reshape(-1, 3)
        return matrix
    
    
    def chek_funtions(self):
        """Функция для проверки, что все необходимые функции переданы в симулятор"""
        
        if self.fields_func is not None:
            print("Функция рассчета полей получена!")
            
        else:
            print("Функция рассчета полей еще не назначена!")
            raise
            
        pass
    
    def simulate(self,
                 V_0, 
                 X_0,
                 q_vector,
                 m_rest,
                 time_sim,
                 delta_time = 5e-10):
        
        self.chek_funtions() # Проверка наличия необходимых функций (Функция для рассчета полей должна быть передана экз. класса перед началом симуляции)

        sim = bdsim.BDSim()  # create simulator
        bd = sim.blockdiagram()  # create an empty block diagram

        # s = time_sim
        # t_sim = s / V_0[0]
        # delta_time = time_sim / 1000

        V_0 = self.tensor_to_vector(V_0)
        X_0 = self.tensor_to_vector(X_0)
        
        # print('\nV_0', V_0)
        # print('\nX_0', X_0)
        
        velocity = bd.INTEGRATOR(V_0)
        coordinate = bd.INTEGRATOR(X_0)
        
        acceleration = bd.FUNCTION(self.acceleration_vector, nin=2, nout=1)
        relativ_mass = bd.FUNCTION(self.relativistic_mass_electron, nin=2, nout=1)
        fields = bd.FUNCTION(self.fields_func.calculate, nin=1, nout=2)
        lorets_forse_block = bd.FUNCTION(self.lorentz_force, nin=3, nout=1)
        coulomb_force_block = bd.FUNCTION(self.coulomb_force, nin=2, nout=1)
        
        acceleration_tensor_to_vector = bd.FUNCTION(self.tensor_to_vector, nin=1, nout=1)
        velocity_vector_to_tensor = bd.FUNCTION(self.vector_to_tensor, nin=1, nout=1)
        coords_vector_to_tensor = bd.FUNCTION(self.vector_to_tensor, nin=1, nout=1)
        
        q_block = bd.CONSTANT(q_vector)
        rest_mass_block = bd.CONSTANT(m_rest)
        
        sum_of_forses = bd.SUM()
        
        # stop_condition = bd.STOP()
        
        # reshaping blocks 
        bd.connect(acceleration, acceleration_tensor_to_vector)
        
        bd.connect(velocity, velocity_vector_to_tensor)
        
        bd.connect(coordinate, coords_vector_to_tensor)
        bd.connect(coords_vector_to_tensor, fields)
        
        
        # connect the blocks of integrators
        bd.connect(acceleration_tensor_to_vector, velocity)
        bd.connect(velocity, coordinate)
        
        # def relativistic_mass_electron(self, v: np.array, m: np.array) -> float:
        bd.connect(velocity_vector_to_tensor, relativ_mass[0])
        bd.connect(rest_mass_block, relativ_mass[1])
        
        # def lorentz_force(self, q, v, b_field) -> np.array:
        bd.connect(q_block, lorets_forse_block[0])
        bd.connect(velocity_vector_to_tensor, lorets_forse_block[1])
        bd.connect(fields[0], lorets_forse_block[2])
        
        # def coulomb_force(self, q, e_field) -> np.array:
        bd.connect(q_block, coulomb_force_block[0])
        bd.connect(fields[1], coulomb_force_block[1])
        
        # summator of forses
        bd.connect(coulomb_force_block, sum_of_forses[0])
        bd.connect(lorets_forse_block, sum_of_forses[1])

        # def acceleration_vector(self, forse, masses) -> np.array:
        bd.connect(sum_of_forses, acceleration[0])
        bd.connect(relativ_mass, acceleration[1])
        
        # velocity
        # bd.connect(acceleration, velocity)
        
        
        # check the diagram
        bd.compile()
        sim.report(bd)   # list the system
        sim.report(bd, type="lists")
        
        # sim.run(bd, tlen=10, dt=0.01) 
        
        
        """autosummary::
        :toctree: generated/

        solve_ivp     -- Convenient function for ODE integration.
        RK23          -- Explicit Runge-Kutta solver of order 3(2).
        RK45          -- Explicit Runge-Kutta solver of order 5(4).
        DOP853        -- Explicit Runge-Kutta solver of order 8.
        Radau         -- Implicit Runge-Kutta solver of order 5.
        BDF           -- Implicit multi-step variable order (1 to 5) solver.
        L.SODA         -- LSODA solver from ODEPACK Fortran package.
        OdeSolver     -- Base class for ODE solvers.
        DenseOutput   -- Local interpolant for computing a dense output.
        OdeSolution   -- Class which represents a continuous ODE solution.
        """

        out = sim.run(bd, time_sim, dt=delta_time, solver="RK45", minstepsize=delta_time/1e20, watch=[acceleration, velocity_vector_to_tensor, coords_vector_to_tensor, fields[0]])

        # print('Отклонение составило ',57.2958 * np.arctan(out.y1[-1][1]/out.y1[-1][0]), "градусов")
        print('Шаг интегрирования', delta_time, ' секунд')
            
            
        return out
    
    
class Colider_simulator_visualization():
    
    def __init__(self, configuration):
        self.config = configuration
        self.trajectories = None
        
        self.lables_size = 6
        
        self.unic_models_dict = {}
        self.all_models_list = []
        
        pass
    
    
    def show_all(self):
        
        # print('\n', self.config.df_agregats_config_id)
        # print('\n', self.config.df_agregats_types)
        
        
        pl = pv.Plotter()
        pl.enable_parallel_projection()
        opacity_comon = 0.2
        color_1 = 'cyan' # "silver" 
        
        
        pv.global_theme.font.size = self.lables_size
        pv.global_theme.font.title_size = self.lables_size
        
        
        # Добавим в график 3д модели магнитов
        for index, row in self.config.df_agregats_config_id.iterrows():
            # try:
            field_id = row['field_id']
            
            apparat_row = self.config.df_agregats_types[self.config.df_agregats_types['field_id'] == field_id]
            # print('apparat_row', apparat_row)
            # print('apparat_row type', type(apparat_row))
            
            directory = apparat_row['directory_of_3D_models'].values[0]
            model_name = apparat_row['3D_model_file_name'].values[0]
        
            
            key_name = directory.replace('\\', os.sep) +"\\".replace('\\', os.sep) + model_name 
            
            # print('\n', 'directory', directory)
            # print('\n', 'directory type', type(directory))

            
            if key_name in self.unic_models_dict:
                # print("Ключ найден!")
                self.all_models_list.append(self.unic_models_dict[key_name])
                model_i = self.unic_models_dict[key_name]
                
                
            else:
                # print("Такого ключа нет.")
                model_i = pv.read(key_name)
                
                self.unic_models_dict[key_name] = model_i
                self.all_models_list.append(self.unic_models_dict[key_name])
                
            
            model_i_coords = np.array([row['x'], row['y'], row['z']]) * 1000 
            alpha, betta, gamma = row['alfa'], row['betta'], row['gamma'] 
            
            # print('\n')
            # print('model_i_coords', model_i_coords)
            # print('alpha, betta, gamma', alpha, betta, gamma)
            
            
            model_i = model_i.rotate_x(alpha, point=[0, 0, 0], inplace=False)
            model_i = model_i.rotate_y(betta, point=[0, 0, 0], inplace=False)
            model_i = model_i.rotate_z(gamma, point=[0, 0, 0], inplace=False)
            
            # model_i.rotate_x(betta, point=[0, 0, 0], inplace=False)
            # model_i.rotate_y(alpha, point=[0, 0, 0], inplace=False)
            # model_i.rotate_z(gamma, point=[0, 0, 0], inplace=False)
            
            model_i.translate(model_i_coords, inplace=True) 
            
            pl.add_mesh(model_i, color=color_1, opacity=opacity_comon, show_edges=True, smooth_shading=True)
            
            # except:
            #     print('Произошла ошибка при попытке загрузить 3д модель. field_id - ', field_id)
            
        
        # print('\n',' self.trajectories.shape',  self.trajectories.shape)
        # print('\n',' self.trajectories',  self.trajectories)
        
        # Добавим имеющиеся траектории частиц
        try:
            for i in range(self.trajectories.shape[1]):
                traektory_i = self.trajectories[:,i,:] * 1000
                
                # print('traektory_i', traektory_i)
                # print('traektory_i sh', traektory_i.shape)
                
                
                path = pv.PolyData(traektory_i)

                # 2. Создаем индексы связей (lines)
                # Формат: [количество_точек, id1, id2, ..., idN]
                n_points = traektory_i.shape[0]
                cells = np.full((n_points - 1, 3), 2, dtype=np.int_)
                cells[:, 1] = np.arange(0, n_points - 1)
                cells[:, 2] = np.arange(1, n_points)

                path.lines = cells

                pl.add_mesh(path, color="blue", line_width=3, label="Траектория " + str(1))
                
        except:
            print('Ошибка вывода траекторий!')

                
        
        pl.add_axes() # Добавляет цветные стрелки XYZ в углу 
        # pl.add_legend() 
        pl.show_grid() 
        pl.show() 
            
            
            
            
        pass
    
    
def create_many_pionts(m, l, s, z_value, dz=0, dy=0):
    # Параметры
    # m, l = 5, 10  # Количество точек по X и Y
    # s = 1.0       # Шаг
    # z_value = 0.0

    # 1. Создаем диапазоны так, чтобы 0 был посередине
    # (m-1)*s — это полная ширина сетки по X
    z = (np.arange(m) - (m - 1) / 2) * s + dz
    y = (np.arange(l) - (l - 1) / 2) * s + dy

    # 2. Генерируем сетку
    Z, Y = np.meshgrid(z, y)

    # 3. Собираем в массив nx3
    points = np.column_stack((
        np.full(Z.size, z_value),
        Y.ravel(),
        Z.ravel()
    ))

    return points


def beam_spread(outy2):
    """Находит максимальное расстояние между частицами в пучке"""
    
    # print('\n', 'outy2[-1]', outy2[-1])
    
    max_dist = pdist(outy2[-1]).max()

    # print(f"Наибольшее расстояние: {max_dist}")
    
    return max_dist
    
    
    
    
    






