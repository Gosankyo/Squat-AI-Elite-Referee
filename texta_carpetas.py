import os

ruta = r"C:\Users\Anxo\Desktop\Proyecto final 4\archive (3)\Squat_Data\Squat_Data"

print(f"--- ANALIZANDO: {ruta} ---")

for root, dirs, files in os.walk(ruta):
    # Si encuentra archivos, dinos cuántos y en qué carpeta
    if files:
        print(f"Carpeta: {root}")
        print(f"  Contiene {len(files)} archivos. Ejemplo: {files[0]}")
        break # Solo necesitamos ver la primera que tenga algo
else:
    print("No se encontró NI UN SOLO archivo en ninguna subcarpeta.")