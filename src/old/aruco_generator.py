#!/usr/bin/env python3
import cv2
import numpy as np

# Seleziona il dizionario
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Genera un marker (ID 23 come esempio)
marker_id = 23
marker_size = 200  # pixel
marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)

# Salva l'immagine
cv2.imwrite(f"marker_{marker_id}.png", marker_image)
print(f"Marker {marker_id} salvato come marker_{marker_id}.png")