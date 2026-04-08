import os
import requests
import time

# Configuramos las URLs base 
URL_BASE_BAD = "https://openpowerlifting.gitlab.io/depthcaptcha/images/bad/"
URL_BASE_GOOD = "https://openpowerlifting.gitlab.io/depthcaptcha/images/good/"

# Carpetas locales donde guardaremos las imágenes
CARPETA_DATASET = "dataset_sentadillas"
CARPETA_BAD = os.path.join(CARPETA_DATASET, "nulas")
CARPETA_GOOD = os.path.join(CARPETA_DATASET, "validas")

# Creamos las carpetas si no existen
os.makedirs(CARPETA_BAD, exist_ok=True)
os.makedirs(CARPETA_GOOD, exist_ok=True)

def descargar_imagenes(url_base, carpeta_destino, etiqueta, max_intentos=500):
    print(f"\n--- Iniciando descarga de imagenes {etiqueta.upper()} ---")
    consecutivos_vacios = 0
    
    # Bucle del 1 al max_intentos (ej: 0001.jpg a 0500.jpg)
    for i in range(1, max_intentos + 1):
        # Formateamos el número para que tenga 4 dígitos: 1 -> "0001"
        nombre_archivo = f"{i:04d}.jpg" 
        url_imagen = f"{url_base}{nombre_archivo}"
        ruta_local = os.path.join(carpeta_destino, nombre_archivo)
        
        # Si ya la descargaste antes, se la salta
        if os.path.exists(ruta_local):
            continue
            
        try:
            respuesta = requests.get(url_imagen)
            
            # Si la imagen existe (código 200), la guardamos
            if respuesta.status_code == 200:
                with open(ruta_local, 'wb') as f:
                    f.write(respuesta.content)
                print(f"[OK] Descargada: {etiqueta}/{nombre_archivo}")
                consecutivos_vacios = 0 # Reseteamos el contador de fallos
            else:
                # Si da error 404 (no existe), sumamos al contador
                consecutivos_vacios += 1
                
            # Si hay 5 errores seguidos, asumimos que ya no hay más imágenes y paramos
            if consecutivos_vacios >= 5:
                print(f"[FIN] No se encontraron mas imagenes en la carpeta {etiqueta} tras {i} intentos.")
                break
                
            # Pequeña pausa para no saturar el servidor
            time.sleep(0.2)
            
        except Exception as e:
            print(f"[ERROR] Fallo en la conexion con {nombre_archivo}: {e}")

if __name__ == "__main__":
    # Descargamos las imágenes
    descargar_imagenes(URL_BASE_BAD, CARPETA_BAD, "nulas", max_intentos=500)
    descargar_imagenes(URL_BASE_GOOD, CARPETA_GOOD, "validas", max_intentos=500)
    print("\n[EXITO] Dataset descargado y clasificado con exito!")