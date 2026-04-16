
import os
import re

import numpy as np
import pandas as pd

from scipy.interpolate import interpn


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
        path = self.all_information['path_to_the_after_processing_file'] + '\\'
        
        return [path + arguments, path + shapes, path + field, path + coordinates, path]
    
    def save_field(self):
        
        names_for_open = self.nawes_of_ready_files()
        
        shapes = np.array(self.argument_field.shape)
        
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
            
            for_open = self.init_files_way + '\\' + file
            
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
        
        # Массив с коэффициентами симметрии. Три коэффициента пространства и один коэффициент симметрии поля. 
        symetric_array = [self.all_information['x_symmetry'], self.all_information['y_symmetry'], 
                                self.all_information['z_symmetry'], self.all_information['I_symmetry']]
        
        # Массив с границами допустимых значенениях параметров. Координаты и значения тока. Границы не учитывают симметрию. 
        coordinates_info_last_I = [self.coordinates_info_last[0], self.coordinates_info_last[1],
                                 self.coordinates_info_last[2], 0.0,
                                 self.coordinates_info_last[3], self.coordinates_info_last[4],
                                 self.coordinates_info_last[5], self.argument_list_sorted[-1]]
        
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


        """Создаем сетку с координатами"""
        x_coords = np.linspace(image_borders[0], image_borders[1], self.field.shape[1])  # начало, конец, кол-во точек
        y_coords = np.linspace(image_borders[2], image_borders[3], self.field.shape[2])
        z_coords = np.linspace(image_borders[4], image_borders[5], self.field.shape[3])
        points = (x_coords, y_coords, z_coords)
        
        self.points = points
 
        # print('\n', 'points', points, '\n')
        # print('\n', 'self.field_shape', self.field.shape, '\n')
        # print('\n', 'self.step', self.step, '\n')
        # print('\n', 'self.real_borders', self.real_borders, '\n')
        # print('\n', 'self.symetric_array', self.symetric_array, '\n')
        # print('\n', 'self.coordinates_info_last_I', self.coordinates_info_last_I, '\n')
        # print('\n', 'image_borders', self.image_borders)



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
        
        
        self.field = self.field.reshape(self.shapes.astype(int))
        self.argument_list_sorted = self.argument_list.copy()
        self.argument_list_sorted = np.sort(self.argument_list_sorted)
        
        if self.enable_messages:    # Переменная, разрешающая вывод сообщений, облегчающих отладку. 
            print('\n', 'магнитное поле', self.field[0][0][0][0], '\n')
            print('\n', 'магнитное поле', self.field[1][0][0][0], '\n')
            print('\n', 'магнитное поле', self.field[2][0][0][0], '\n')
            print('\n', 'магнитное поле', self.field[3][0][0][0], '\n')
            print('\n', 'магнитное поле', self.field[4][0][0][0], '\n')
        
        
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
    
    def find_real_args(self, argument: float):
        """Если аргумент выходит за пределеы дискретно заданной сетки (току больше, чем было измерено во 
        время численного моделирования), то поле будет рассчитываться на основе самого большого известного поля"""
        max_arg = self.argument_list_sorted[-1]
        if max_arg <= argument:
             argument = max_arg
        
        return argument
    
    def find_intermediate_vector(self, vector_1, vector_2, step, local_argument):
        """Функция получает два вектоа, шаг между ними и промежуточное значение 
        а возвращает промежуточный вектор между двумя данными. 
        Подразумевается, что первый вектор расположен левее на оси координат. А локальный 
        аргумент - расстояние между левым вектором и искомым."""
        
        vector_new = ((vector_2 - vector_1)/step) * local_argument + vector_1
        
        return vector_new
    
    def find_two_closest_tensors(self, argument):
        """Определяем два длижайших по аргументу тензоров (наиболее близкие токи)"""

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
        
        # print('\n', 'self.points, self.field[argument_index], argument')
        # print(self.points, self.field[argument_index].shape, argument)
        
        res = interpn(self.points, self.field[argument_index], self.real_x_coordinates, method=self.method)
        
        res = np.array(res)[0]
        
        return res
        
        
    def start_approximation(self, argument, in_real_borders, in_arg_bord):
        
        if self.enable_messages:
            print('\n', 'self.step', self.step, '\n')
        
        argument_index_min, argument_index_big = self.find_two_closest_tensors(argument)
        
        # print('\n', 'argument_index_min, argument_index_big', argument_index_min, argument_index_big )
        
        if argument_index_min == argument_index_big:
            vector_third = self.find_vector_by_coords_in_tensor(argument, argument_index_min)
            
        else:
            vector_first = self.find_vector_by_coords_in_tensor(argument, argument_index_min)
            vector_secoen = self.find_vector_by_coords_in_tensor(argument, argument_index_big)
            
            # print('\n')
            # print('\n', 'vector_first', vector_first)
            # print('\n', 'vector_secoen', vector_secoen)
            
            vector_third = self.find_intermediate_vector(vector_first, vector_secoen , self.step_arg, self.local_arg_amper)
            
            # print('\n', 'vector_third', vector_third)
            
            vector_third = vector_third * self.sumetric_coeffs[-1]
            
        vector_third = vector_third * self.sumetric_coeffs[:-1]
        self.vector_third = vector_third
        
        if self.enable_messages:
            print('\n', 'self.sumetric_coeffs', self.sumetric_coeffs, '\n')
            print('\n', 'vector_third', vector_third, '\n')
        
        return vector_third
    
    
    def calculate(self, x_coordinates: np.array, argument: float, method = 'linear') -> np.array:
        """Функция, рассчитывающая магнитное поле при произвольном токе и при произвольной координате, 
        в рамках заданного диапазона. Если ток выйдет за пределы имеющегося диапазона, метод вернет предупреждение
        и вектор, рассчитанный на экстрополяции ближайших известных значений апроксимации.
        Если координаты выйдут за пределы установленного диапазона, функция вернет нулевой вектор."""
        
        self.method = method
        
        in_real_borders, in_arg_bord = self.check_borders(x_coordinates, argument, self.real_borders) # Проверяем, возможно ли рассчитать поле для заявленных координат 
        check = in_real_borders.prod()
        self.x_coordinates = x_coordinates
        
         
        if not(check):
            """Если координата находится за пределами фактических границ поля, функция позвращает нулевой вектор"""
            if self.enable_messages:
                print('\n', 'Запрашиваемые координаты за пределами дискретно заданного поля', '\n')
            # print('\n', self.x_coordinates)
            return np.array([0.0, 0.0, 0.0])
        
        else:
            """В противном случае программа перерассчитывает координаты с учетом симметрии, сохраняет коэффициенты и
            определяет координаты восьми ближайших известных векторов, чтобы приступить к билинейной интерполяции"""
            if self.enable_messages:
                print('\n', 'Запрашиваемые координаты находятся в пределах дискретно заданного поля', '\n')
        
            self.find_real_coords(x_coordinates, argument)
            
            argument = self.find_real_args(argument)
            vector_third = self.start_approximation(argument, in_real_borders, in_arg_bord)
            return vector_third
            
        pass
