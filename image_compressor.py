import numpy as np
import cv2
from scipy.fftpack import dct, idct
import pywt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image


class ImageCompressor:
    
    def __init__(self):
        self.original_image = None
        self.compressed_dct = None
        self.compressed_dwt = None
        
    @staticmethod
    def psnr(original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr_value
    
    @staticmethod
    def dct2(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    @staticmethod
    def idct2(block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def compress_dct(self, image, quality_factor=50):
       
        h, w = image.shape
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        
        if h_pad > 0 or w_pad > 0:
            image = np.pad(image, ((0, h_pad), (0, w_pad)), mode='edge')
        
        h, w = image.shape
        
        quantization_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        
        if quality_factor < 50:
            scale = 5000 / quality_factor
        else:
            scale = 200 - 2 * quality_factor
        scale = scale / 100
        
        quantization_matrix = np.floor((quantization_matrix * scale + 50) / 100)
        quantization_matrix[quantization_matrix == 0] = 1
        
        compressed = np.zeros_like(image, dtype=float)
        
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = image[i:i+8, j:j+8].astype(float)
                
                dct_block = self.dct2(block)
                
                quantized = np.round(dct_block / quantization_matrix)
                
                dequantized = quantized * quantization_matrix
                
                compressed[i:i+8, j:j+8] = self.idct2(dequantized)
        
        compressed = compressed[:h-h_pad, :w-w_pad]
        
        compressed = np.clip(compressed, 0, 255)
        
        return compressed.astype(np.uint8)
    
    def compress_dwt(self, image, wavelet='haar', threshold_percent=10):
        coeffs = pywt.wavedec2(image.astype(float), wavelet, level=3)
       
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        
        threshold = np.percentile(np.abs(coeff_arr), 100 - threshold_percent)
        
        coeff_arr_thresh = pywt.threshold(coeff_arr, threshold, mode='soft')
        
        coeffs_thresh = pywt.array_to_coeffs(coeff_arr_thresh, coeff_slices, output_format='wavedec2')
        compressed = pywt.waverec2(coeffs_thresh, wavelet)
        compressed = compressed[:image.shape[0], :image.shape[1]]
        
        compressed = np.clip(compressed, 0, 255)
        
        return compressed.astype(np.uint8)
    
    def load_image(self, filepath):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Не вдалося завантажити зображення")
        self.original_image = image
        return image
    
    def compress_image(self, dct_quality=50, dwt_threshold=10):
       
        if self.original_image is None:
            raise ValueError("Спочатку завантажте зображення")
        
        self.compressed_dct = self.compress_dct(self.original_image, dct_quality)
        psnr_dct = self.psnr(self.original_image, self.compressed_dct)
        
        self.compressed_dwt = self.compress_dwt(self.original_image, threshold_percent=dwt_threshold)
        psnr_dwt = self.psnr(self.original_image, self.compressed_dwt)
        
        return self.compressed_dct, self.compressed_dwt, psnr_dct, psnr_dwt


class CompressorGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Стискувач зображень DCT/DWT")
        self.root.geometry("1400x800")
        
        self.compressor = ImageCompressor()
        self.current_image_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(top_frame, text="Завантажити зображення", 
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Стиснути", 
                  command=self.compress).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Зберегти результати", 
                  command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        control_frame = ttk.LabelFrame(self.root, text="Параметри стиснення", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="DCT Якість (0-100):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.dct_quality = tk.IntVar(value=50)
        dct_slider = ttk.Scale(control_frame, from_=1, to=100, variable=self.dct_quality, 
                              orient=tk.HORIZONTAL, length=300)
        dct_slider.grid(row=0, column=1, padx=5, pady=5)
        self.dct_label = ttk.Label(control_frame, text="50")
        self.dct_label.grid(row=0, column=2, padx=5, pady=5)
        dct_slider.config(command=lambda v: self.dct_label.config(text=f"{int(float(v))}"))
        
        ttk.Label(control_frame, text="DWT Коефіцієнти (0-100):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.dwt_threshold = tk.IntVar(value=10)
        dwt_slider = ttk.Scale(control_frame, from_=1, to=100, variable=self.dwt_threshold, 
                              orient=tk.HORIZONTAL, length=300)
        dwt_slider.grid(row=1, column=1, padx=5, pady=5)
        self.dwt_label = ttk.Label(control_frame, text="10")
        self.dwt_label.grid(row=1, column=2, padx=5, pady=5)
        dwt_slider.config(command=lambda v: self.dwt_label.config(text=f"{int(float(v))}"))
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 5))
        self.fig.suptitle('Порівняння методів стиснення зображень', fontsize=14, fontweight='bold')
        
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.status_label = ttk.Label(self.root, text="Готовий до роботи", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_image(self):
        filepath = filedialog.askopenfilename(
            title="Виберіть зображення",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.current_image_path = filepath
                self.compressor.load_image(filepath)
                
                self.axes[0].clear()
                self.axes[0].imshow(self.compressor.original_image, cmap='gray')
                self.axes[0].set_title('Оригінальне зображення')
                self.axes[0].axis('off')
                
                self.axes[1].clear()
                self.axes[1].text(0.5, 0.5, 'Натисніть "Стиснути"', 
                                ha='center', va='center', fontsize=12)
                self.axes[1].axis('off')
                
                self.axes[2].clear()
                self.axes[2].text(0.5, 0.5, 'Натисніть "Стиснути"', 
                                ha='center', va='center', fontsize=12)
                self.axes[2].axis('off')
                
                self.fig.canvas.draw()
                
                self.status_label.config(text=f"Завантажено: {filepath} | Розмір: {self.compressor.original_image.shape}")
                
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося завантажити зображення:\n{str(e)}")
    
    def compress(self):
        if self.compressor.original_image is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте зображення!")
            return
        
        try:
            self.status_label.config(text="Стиснення...")
            self.root.update()
            
            dct_q = self.dct_quality.get()
            dwt_t = self.dwt_threshold.get()
            
            compressed_dct, compressed_dwt, psnr_dct, psnr_dwt = self.compressor.compress_image(
                dct_quality=dct_q,
                dwt_threshold=dwt_t
            )
            
            self.axes[0].clear()
            self.axes[0].imshow(self.compressor.original_image, cmap='gray')
            self.axes[0].set_title('Оригінальне зображення', fontsize=12, fontweight='bold')
            self.axes[0].axis('off')
            
            self.axes[1].clear()
            self.axes[1].imshow(compressed_dct, cmap='gray')
            self.axes[1].set_title(f'DCT Стиснення\nPSNR: {psnr_dct:.2f} дБ\nЯкість: {dct_q}', 
                                  fontsize=11, fontweight='bold')
            self.axes[1].axis('off')
            
            self.axes[2].clear()
            self.axes[2].imshow(compressed_dwt, cmap='gray')
            self.axes[2].set_title(f'DWT Стиснення (Haar)\nPSNR: {psnr_dwt:.2f} дБ\nКоеф.: {dwt_t}%', 
                                  fontsize=11, fontweight='bold')
            self.axes[2].axis('off')
            
            self.fig.tight_layout()
            self.fig.canvas.draw()
            
            self.status_label.config(text=f"Готово! DCT PSNR: {psnr_dct:.2f} дБ | DWT PSNR: {psnr_dwt:.2f} дБ")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при стисненні:\n{str(e)}")
            self.status_label.config(text="Помилка при стисненні")
    
    def save_results(self):
        if self.compressor.compressed_dct is None or self.compressor.compressed_dwt is None:
            messagebox.showwarning("Попередження", "Спочатку виконайте стиснення!")
            return
        
        try:
            directory = filedialog.askdirectory(title="Виберіть папку для збереження")
            if directory:
                cv2.imwrite(f"{directory}/compressed_dct.png", self.compressor.compressed_dct)
                
                cv2.imwrite(f"{directory}/compressed_dwt.png", self.compressor.compressed_dwt)
                
                self.fig.savefig(f"{directory}/comparison.png", dpi=150, bbox_inches='tight')
                
                messagebox.showinfo("Успіх", f"Результати збережено в:\n{directory}")
                self.status_label.config(text=f"Результати збережено в: {directory}")
                
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося зберегти результати:\n{str(e)}")


def main():
    root = tk.Tk()
    app = CompressorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

