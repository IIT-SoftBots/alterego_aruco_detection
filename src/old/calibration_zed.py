#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

def draw_ui(frame, images_captured, max_images, chessboard_found):
    """Disegna l'interfaccia utente sul frame"""
    # Disegna un rettangolo semi-trasparente per il testo
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Mostra il conteggio delle immagini
    cv2.putText(frame, 
                f"Immagini: {images_captured}/{max_images}", 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2)
    
    # Mostra lo stato della rilevazione
    status_color = (0, 255, 0) if chessboard_found else (0, 0, 255)
    status_text = "Scacchiera Rilevata" if chessboard_found else "Scacchiera NON Rilevata"
    cv2.putText(frame, 
                status_text, 
                (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                status_color, 
                2)
    
    # Mostra le istruzioni
    cv2.putText(frame, 
                "Click sinistro = Salva  |  Click destro = Esci", 
                (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2)

def mouse_callback(event, x, y, flags, param):
    """Gestisce gli eventi del mouse"""
    if event == cv2.EVENT_LBUTTONDOWN:  # Click sinistro
        param['capture'] = True
    elif event == cv2.EVENT_RBUTTONDOWN:  # Click destro
        param['quit'] = True

def calibrate_camera():
    CHECKERBOARD = (7,7)
    
    # Prepara i punti oggetto
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
    
    square_size = 0.02  # 2cm
    objp = objp * square_size
    
    objpoints = []
    imgpoints = []
    
    # Inizializza la camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Crea la finestra e imposta il callback del mouse
    cv2.namedWindow('Calibrazione')
    param = {'capture': False, 'quit': False}
    cv2.setMouseCallback('Calibrazione', mouse_callback, param)
    
    images_captured = 0
    max_images = 20
    
    while images_captured < max_images and not param['quit']:
        ret, frame = cap.read()
        if not ret:
            print("Errore nella cattura del frame")
            continue
        
        # Usa solo la metà destra per ZED
        height, width = frame.shape[:2]
        mid_width = width // 2
        frame = frame[:, 0:mid_width]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        display_frame = frame.copy()
        
        if ret:
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret)
            
            if param['capture']:  # Se è stato fatto click sinistro
                objpoints.append(objp)
                imgpoints.append(corners)
                images_captured += 1
                print(f"Immagine {images_captured}/{max_images} catturata")
                param['capture'] = False  # Reset flag
        
        # Aggiorna UI
        draw_ui(display_frame, images_captured, max_images, ret)
        cv2.imshow('Calibrazione', display_frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Procedi con la calibrazione se abbiamo abbastanza immagini
    if images_captured > 0:
        print("Calibrazione in corso...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        print("\nCamera Matrix:")
        print(mtx)
        print("\nDistortion Coefficients:")
        print(dist)
        
        np.savez('calibration_data.npz', 
                 camera_matrix=mtx,
                 dist_coeffs=dist)
        
        print("\nParametri di calibrazione salvati in 'calibration_data.npz'")
        
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print(f"\nErrore di riproiezione totale: {mean_error/len(objpoints)}")

if __name__ == '__main__':
    calibrate_camera()