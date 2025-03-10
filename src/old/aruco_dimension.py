from PIL import Image

# Carica l'immagine generata
marker_id = 23
img = Image.open(f"marker_{marker_id}.png")

# Imposta la dimensione corretta in centimetri e DPI
dpi = 300
cm_to_inches = 20 / 2.54  # 20 cm convertiti in pollici
new_size = (int(dpi * cm_to_inches), int(dpi * cm_to_inches))  # 2362 × 2362 pixel

# Ridimensiona e salva con DPI 300
img = img.resize(new_size, Image.Resampling.LANCZOS)
img.save(f"marker_{marker_id}_300dpi_20cm.png", dpi=(dpi, dpi))

print(f"Marker salvato come marker_{marker_id}_300dpi_20cm.png con 300 DPI e dimensioni 20x20 cm.")
