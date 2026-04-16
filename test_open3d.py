import open3d as o3d
import numpy as np

# 1. Создаем первый объект (сфера)
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
sphere.compute_vertex_normals()
sphere.paint_uniform_color([0.1, 0.7, 0.1]) # Зеленый цвет

# 2. Создаем второй объект (координатные оси)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

# 3. Сдвигаем сферу, чтобы объекты не перекрывали друг друга
sphere.translate([3, 0, 0])

# 4. Визуализируем оба объекта, передав их списком
o3d.visualization.draw_geometries([sphere, axes], 
                                  window_name="Open3D Scene",
                                  width=800, 
                                  height=600)
