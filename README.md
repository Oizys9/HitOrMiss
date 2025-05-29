# HitOrMiss
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load gambar biner (hitam putih)
img = cv2.imread('karakter.png', cv2.IMREAD_GRAYSCALE)
_, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Fungsi Hit-or-Miss
def hit_or_miss(img_bin, se_foreground, se_background):
    # Langkah 1: komplemen gambar (foreground menjadi background)
    img_comp = cv2.bitwise_not(img_bin)

    # Langkah 2: erosi gambar asli dengan SE foreground
    eroded_fg = cv2.erode(img_bin, se_foreground)

    # Langkah 3: erosi komplemen gambar dengan SE background
    eroded_bg = cv2.erode(img_comp, se_background)

    # Langkah 4: Interseksi (AND) dari dua hasil erosi
    hitmiss = cv2.bitwise_and(eroded_fg, eroded_bg)

    return hitmiss

# Contoh SE untuk mendeteksi sudut kanan atas (3x3)
# 1 = foreground (harus ada)
# 0 = background (harus tidak ada)
# -1 = don't care (abaikan)

# Kita buat dua SE untuk foreground dan background

# SE foreground (hit)
se_fg = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [0, 0, 1]], dtype=np.uint8)

# SE background (miss)
se_bg = np.array([[1, 1, 1],
                  [1, 0, 0],
                  [1, 1, 0]], dtype=np.uint8)

# Karena OpenCV erosi hanya menerima 0 dan 1 (uint8), kita perlu ubah -1 jadi 0 di SE,
# dan untuk 'don't care', kita tidak memasukkan pixel tersebut ke erosi,
# tapi kita akan menggunakan trik dengan masking:

def prepare_se(se):
    """Mengubah -1 jadi 0 dan mengembalikan mask untuk don't care."""
    mask = (se != 2).astype(np.uint8)  # in case 2 is used for don't care
    se_fixed = se.copy()
    se_fixed[se_fixed == -1] = 0
    return se_fixed, mask

# Namun OpenCV erode tidak mendukung mask, jadi trik standar:
# Buat dua erosi terpisah, satu untuk foreground dengan SE foreground,
# satu untuk background dengan SE background,
# dan lakukan AND.

# Jadi kita gunakan se_fg dan se_bg sebagai binary mask (1 dan 0),
# agar cocok dengan OpenCV.

# Pastikan se_fg dan se_bg bertipe uint8 dan hanya 0/1
se_fg = (se_fg == 1).astype(np.uint8)
se_bg = (se_bg == 1).astype(np.uint8)

# Terapkan Hit-or-Miss
result = hit_or_miss(img_bin, se_fg, se_bg)

# Tampilkan hasil
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Gambar Asli")
plt.imshow(img_bin, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Hit or Miss")
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Overlay Hit or Miss")
overlay = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
overlay[result > 0] = [255, 0, 0]  # warna merah untuk hit or miss
plt.imshow(overlay)
plt.axis('off')

plt.show()
