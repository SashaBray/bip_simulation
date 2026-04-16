import unittest
import numpy as np
import numpy.testing as npt

from colider_simulation_instruments_lib import *


magnet_device = Field_generation_device()

print('magnet_device type', type(magnet_device))

# if isinstance(magnet_device, Field_generation_device):
#     print("***")
#     pass


class TestMyFunction(unittest.TestCase):
    def test_get_some_data(self):
        
        simple_vector = np.array([1, 0, 0])
                
        input, expected = np.array([0, 0, 0]), np.array([1.0, 0.0, 0.0]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        input, expected = np.array([0, 90, 0]), np.array([0.0, 0.0, 1.0]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        input, expected = np.array([90, 0, 0]), np.array([1.0, 0.0, 0.0]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        input, expected = np.array([0, 0, 90]), np.array([0.0, 1.0, 0.0]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        input, expected = np.array([0, -90, 0]), np.array([0.0, 0.0, -1.0]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        input, expected = np.array([-90, 0, 0]), np.array([1.0, 0.0, 0.0]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        input, expected = np.array([0, 0, -90]), np.array([0.0, -1.0, 0.0]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        simple_vector_i, input, expected = np.array([1, 1, 1]), np.array([0, 0, 90]), np.array([-1.0, 1.0, 1.0]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector_i
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        simple_vector_i, input, expected = np.array([1, 1, 0]), np.array([-90, 0, 0]), np.array([1.0, 0.0, -1.0]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector_i        
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        simple_vector_i, input, expected = np.array([1, 0, 0]), np.array([0, 90, 0]), np.array([0, 0, 1]) 
        result = magnet_device.get_direction_cosine_matrix(input) @ simple_vector_i        
        npt.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)
        
        
    # Тест: неверное количество элементов
    def test_wrong_shape(self):
        
        bad_inputs = [np.array([1, 2]), np.array([1, 2, 3, 4]), np.array([])]
        for inp in bad_inputs:
            with self.subTest(inp=inp):
                self.assertRaises(ValueError, magnet_device.get_direction_cosine_matrix, inp)
        
    # Тест: неверный тип содержимого (строки вместо чисел)
    def test_wrong_dtype(self):
        bad_data = np.array(["a", "b", "c"])
        with self.assertRaises(TypeError):
            magnet_device.get_direction_cosine_matrix(bad_data)     
            
    # Тест: передача None или других типов
    def test_invalid_types(self):
        with self.assertRaises(Exception): # Можно уточнить тип ошибки
            magnet_device.get_direction_cosine_matrix(None)

    # Тест: проверка структуры выходных данных (успешный сценарий)
    def test_output_structure(self):
        result = magnet_device.get_direction_cosine_matrix(np.array([0, 0, 0]))
        self.assertEqual(result.shape, (3, 3))
        self.assertIsInstance(result, np.ndarray)   
        
        
    def test_coordinate_system_translation(self):
        """Let's test different coordinates to make sure the function returns the correct values.
        """
        print('**')
        
        simple_vector = np.array([100, 100, 100])
        dergis = np.array([0, 0, 0])
        shift_vector = np.array([100, 100, 100])
        expected = np.array([0.0, 0.0, 0.0]) 
        
        rotation_matrix = magnet_device.get_direction_cosine_matrix(-dergis)
        new_simple_vector = magnet_device.find_local_coordinates(simple_vector, shift_vector, rotation_matrix) 
        npt.assert_allclose(new_simple_vector, expected, atol=1e-7, rtol=1e-7)
        
        
        simple_vector = np.array([101, 100, 100])
        dergis = np.array([0, 0, 90])
        shift_vector = np.array([100, 100, 100])
        expected = np.array([0.0, -1.0, 0.0]) 
        
        rotation_matrix = magnet_device.get_direction_cosine_matrix(-dergis)
        new_simple_vector = magnet_device.find_local_coordinates(simple_vector, shift_vector, rotation_matrix)
        npt.assert_allclose(new_simple_vector, expected, atol=1e-7, rtol=1e-7)
        
        
        simple_vector = np.array([101, 101, 100])
        dergis = np.array([0, 0, 90])
        shift_vector = np.array([100, 100, 100])
        expected = np.array([1.0, -1.0, 0.0]) 
        
        rotation_matrix = magnet_device.get_direction_cosine_matrix(-dergis)
        new_simple_vector = magnet_device.find_local_coordinates(simple_vector, shift_vector, rotation_matrix)
        npt.assert_allclose(new_simple_vector, expected, atol=1e-7, rtol=1e-7)
        
        
        simple_vector = np.array([101, 101, 100]) 
        dergis = np.array([0, 90, 90]) 
        shift_vector = np.array([100, 100, 100]) 
        expected = np.array([1.0, 0.0, -1.0]) 
        
        rotation_matrix = magnet_device.get_direction_cosine_matrix(-dergis)
        new_simple_vector = magnet_device.find_local_coordinates(simple_vector, shift_vector, rotation_matrix)
        npt.assert_allclose(new_simple_vector, expected, atol=1e-7, rtol=1e-7)
        
        
    def test_wrong_dtype_local_system(self):
                
        simple_vector = np.array(["a", "b", "c"])
        shift_vector = np.array([100, 100, 100]) 
        rotation_matrix = np.random.rand(3, 3)
        
        with self.assertRaises(TypeError):
            magnet_device.find_local_coordinates(simple_vector, shift_vector, rotation_matrix)   
            
        simple_vector = np.array([100, 100, 100]) 
        shift_vector = np.array(["a", "b", "c"])
        rotation_matrix = np.random.rand(3, 3)
        
        with self.assertRaises(TypeError):
            magnet_device.find_local_coordinates(simple_vector, shift_vector, rotation_matrix)
            
        simple_vector = np.array([100, 100, 100]) 
        shift_vector = np.array([100, 100, 100]) 
        rotation_matrix = np.array([["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]) 
        
        with self.assertRaises(TypeError):
            magnet_device.find_local_coordinates(simple_vector, shift_vector, rotation_matrix)
            
            
        
        

        pass


 
if __name__ == '__main__':
    unittest.main()