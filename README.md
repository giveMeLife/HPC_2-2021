# HPC_2-2021
Integrantes: Juan Fernández Muñoz
	           Catalina Morales Rojas

Consideraciones: 

- El algoritmo fue implementado con ángulos entre 0° y 180°
- La imagen de salida que tiene el nombre que se ingresa como flag -o corresponde a la obtenida mediante el uso de
  simd luego de aplicar el umbral.
- Además, se genera la imagen obtenida de manera secuencial con nombre secuencial_nombreImagen.raw
- Si se desea obtener la imagen antes de umbralizar, el flag -U debe ser 0
- Para parametrizar la distancia (rho) se utiliza la ecuación 2*(sqrt(x^2 + y^2)) para considerar tanto la parte
  negativa como positiva de las distancias. Luego a ese rango de valores, se divide por R ingresado como parámetro.
- La umbralización transforma todos los valores de la matriz de Hough mayores a U en 255 y los menores en 0.
