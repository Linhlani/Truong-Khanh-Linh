import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, filedialog, Frame
from PIL import Image, ImageTk

# Hàm hiển thị ảnh với tiêu đề
def display_image(image, title, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Hàm dò biên bằng toán tử Sobel
def sobel_edge_detection(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    return sobel_combined

# Hàm dò biên bằng Laplace Gaussian
def laplace_gaussian_edge_detection(image):
    laplacian_gaussian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_gaussian = cv2.normalize(laplacian_gaussian, None, 0, 255, cv2.NORM_MINMAX)
    return laplacian_gaussian

# Hàm xử lý khi chọn ảnh
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    
    # Đọc và hiển thị ảnh đã chọn
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Không thể đọc ảnh. Hãy kiểm tra lại.")
        return
    
    img_display = cv2.cvtColor(cv2.resize(image, (300, 300)), cv2.COLOR_GRAY2RGB)
    img_display = ImageTk.PhotoImage(image=Image.fromarray(img_display))
    img_label.config(image=img_display)
    img_label.image = img_display

    # Thực hiện Sobel và Laplace Gaussian
    sobel_result = sobel_edge_detection(image)
    laplacian_result = laplace_gaussian_edge_detection(image)
    
    # Hiển thị kết quả
    display_image(sobel_result, 'Sobel Combined')
    display_image(laplacian_result, 'Laplace Gaussian')

# Hàm khởi tạo giao diện
def create_gui():
    window = Tk()
    window.title("Dò Biên Ảnh - Sobel và Laplace Gaussian")
    window.geometry("600x500")
    window.config(bg="#f5f5f5")  # Màu nền tinh tế
    
    # Tiêu đề
    title_label = Label(window, text="Dò Biên Ảnh", font=("Arial", 20, "bold"), bg="#f5f5f5")
    title_label.pack(pady=10)
    
    # Khung chứa ảnh
    frame = Frame(window, bg="#ffffff", bd=2, relief="sunken")
    frame.pack(pady=10)
    
    global img_label
    img_label = Label(frame)
    img_label.pack()
    
    # Nút chọn ảnh
    select_button = Button(window, text="Chọn Ảnh", font=("Arial", 14), bg="#007ACC", fg="#ffffff", 
                           command=open_file, padx=20, pady=10)
    select_button.pack(pady=20)
    
    # Chạy GUI
    window.mainloop()

# Gọi hàm khởi tạo GUI
create_gui()
