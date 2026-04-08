import numpy as np
import os

# Ruta a uno de los archivos que detectó tu diagnóstico
archivo_test = r"C:\Users\Anxo\Desktop\Proyecto final 4\archive (3)\Squat_Data\Squat_Data\Invalid\0\0.npy"

try:
    datos = np.load(archivo_test)
    print(f"--- INFO DEL ARCHIVO ---")
    print(f"Forma de los datos (Shape): {datos.shape}")
    print(f"Contenido del primer frame:\n{datos[0]}")
    
    # Si la forma es algo como (numero, 33, 4) o similar, 
    # es que son los landmarks de MediaPipe que el autor ya extrajo.
except Exception as e:
    print(f"Error al cargar: {e}")