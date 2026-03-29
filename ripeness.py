import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

class FruitRipenessDetector:
    def __init__(self):
        self.original_img = None
        self.fruit_mask = None
        self.ripe_mask = None
        self.unripe_mask = None
        self.overripe_mask = None

    def load_image(self, image_path):
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError("Could not load image")
        return self.original_img

    def preprocess_image(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 1.0)
        return blurred

    def segment_fruit_from_background(self, hsv_img):
        h, s, v = cv2.split(hsv_img)
        gray = cv2.cvtColor(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            _, sat_thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, val_thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            combined = cv2.bitwise_and(sat_thresh, val_thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return np.ones(hsv_img.shape[:2], dtype=np.uint8) * 255

        valid_contours = []
        img_area = hsv_img.shape[0] * hsv_img.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < img_area * 0.95 and area > 100:
                valid_contours.append(contour)

        if len(valid_contours) == 0:
            valid_contours = contours

        largest_contour = max(valid_contours, key=cv2.contourArea)
        fruit_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(fruit_mask, [largest_contour], -1, 255, -1)

        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)

        return fruit_mask

    def detect_edges(self, img):
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges

    def segment_ripeness_by_color(self, hsv_img, fruit_mask):
        h, s, v = cv2.split(hsv_img)

        h_masked = cv2.bitwise_and(h, h, mask=fruit_mask)
        s_masked = cv2.bitwise_and(s, s, mask=fruit_mask)
        v_masked = cv2.bitwise_and(v, v, mask=fruit_mask)

        fruit_pixels = fruit_mask > 0
        avg_hue = np.mean(h_masked[fruit_pixels]) if np.sum(fruit_pixels) > 0 else 0
        avg_sat = np.mean(s_masked[fruit_pixels]) if np.sum(fruit_pixels) > 0 else 0
        avg_val = np.mean(v_masked[fruit_pixels]) if np.sum(fruit_pixels) > 0 else 0

        green_pixels = (h_masked >= 40) & (h_masked <= 85) & (s_masked >= 30) & (s_masked <= 200) & (fruit_mask > 0)
        green_pct = np.sum(green_pixels) / np.sum(fruit_pixels) * 100 if np.sum(fruit_pixels) > 0 else 0

        dark_pixels = (v_masked < 100) & (fruit_mask > 0)
        dark_pct = np.sum(dark_pixels) / np.sum(fruit_pixels) * 100 if np.sum(fruit_pixels) > 0 else 0

        low_sat_unripe = avg_sat < 70

        unripe_mask = cv2.bitwise_and(
            cv2.inRange(h_masked, 40, 85),
            cv2.inRange(s_masked, 30, 190)
        )
        unripe_mask = cv2.bitwise_and(unripe_mask, cv2.inRange(v_masked, 80, 255))

        low_sat_light = cv2.bitwise_and(
            cv2.inRange(s_masked, 0, 70),
            cv2.inRange(v_masked, 200, 255)
        )
        unripe_mask = cv2.bitwise_or(unripe_mask, low_sat_light)
        unripe_mask = cv2.bitwise_and(unripe_mask, fruit_mask)

        overripe_dark = cv2.inRange(v_masked, 0, 100)

        overripe_brown = cv2.bitwise_and(
            cv2.inRange(h_masked, 5, 35),
            cv2.inRange(s_masked, 40, 255)
        )
        overripe_brown = cv2.bitwise_and(overripe_brown, cv2.inRange(v_masked, 30, 220))

        overripe_mask = cv2.bitwise_or(overripe_dark, overripe_brown)
        overripe_mask = cv2.bitwise_and(overripe_mask, fruit_mask)

        if low_sat_unripe and green_pct > 30:
            overripe_mask = cv2.bitwise_and(overripe_mask, cv2.bitwise_not(unripe_mask))

        unripe_mask = cv2.bitwise_and(unripe_mask, cv2.bitwise_not(overripe_mask))

        ripe_red1 = cv2.bitwise_and(
            cv2.inRange(h_masked, 0, 10),
            cv2.inRange(s_masked, 70, 255)
        )
        ripe_red1 = cv2.bitwise_and(ripe_red1, cv2.inRange(v_masked, 100, 255))

        ripe_red2 = cv2.bitwise_and(
            cv2.inRange(h_masked, 170, 180),
            cv2.inRange(s_masked, 70, 255)
        )
        ripe_red2 = cv2.bitwise_and(ripe_red2, cv2.inRange(v_masked, 100, 255))

        ripe_yellow = cv2.bitwise_and(
            cv2.inRange(h_masked, 15, 50),
            cv2.inRange(s_masked, 120, 255)
        )
        ripe_yellow = cv2.bitwise_and(ripe_yellow, cv2.inRange(v_masked, 140, 255))

        if green_pct > 40 and low_sat_unripe:
            ripe_yellow = np.zeros_like(ripe_yellow)

        ripe_mask = cv2.bitwise_or(ripe_red1, ripe_red2)
        ripe_mask = cv2.bitwise_or(ripe_mask, ripe_yellow)
        ripe_mask = cv2.bitwise_and(ripe_mask, fruit_mask)
        ripe_mask = cv2.bitwise_and(ripe_mask, cv2.bitwise_not(overripe_mask))
        ripe_mask = cv2.bitwise_and(ripe_mask, cv2.bitwise_not(unripe_mask))

        classified = cv2.bitwise_or(cv2.bitwise_or(unripe_mask, ripe_mask), overripe_mask)
        unclassified = cv2.bitwise_and(fruit_mask, cv2.bitwise_not(classified))

        if np.sum(unclassified) > 0:
            unc_pixels = unclassified > 0
            avg_unc_hue = np.mean(h_masked[unc_pixels]) if np.sum(unc_pixels) > 0 else 90
            avg_unc_sat = np.mean(s_masked[unc_pixels]) if np.sum(unc_pixels) > 0 else 128
            avg_unc_val = np.mean(v_masked[unc_pixels]) if np.sum(unc_pixels) > 0 else 128

            if low_sat_unripe and green_pct > 30:
                unripe_mask = cv2.bitwise_or(unripe_mask, unclassified)
            elif 40 <= avg_unc_hue <= 85 and avg_unc_sat < 190:
                unripe_mask = cv2.bitwise_or(unripe_mask, unclassified)
            elif (avg_unc_hue <= 10 or avg_unc_hue >= 170) and avg_unc_sat > 60:
                ripe_mask = cv2.bitwise_or(ripe_mask, unclassified)
            elif 15 <= avg_unc_hue <= 50 and avg_unc_sat > 100 and green_pct < 25:
                ripe_mask = cv2.bitwise_or(ripe_mask, unclassified)
            elif avg_unc_val < 90 and avg_unc_sat > 60:
                overripe_mask = cv2.bitwise_or(overripe_mask, unclassified)
            else:
                areas = [np.sum(unripe_mask), np.sum(ripe_mask), np.sum(overripe_mask)]
                max_idx = np.argmax(areas)
                if max_idx == 0:
                    unripe_mask = cv2.bitwise_or(unripe_mask, unclassified)
                elif max_idx == 1:
                    ripe_mask = cv2.bitwise_or(ripe_mask, unclassified)
                else:
                    overripe_mask = cv2.bitwise_or(overripe_mask, unclassified)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        unripe_mask = cv2.morphologyEx(unripe_mask, cv2.MORPH_OPEN, kernel)
        ripe_mask = cv2.morphologyEx(ripe_mask, cv2.MORPH_OPEN, kernel)
        overripe_mask = cv2.morphologyEx(overripe_mask, cv2.MORPH_CLOSE, kernel)

        return unripe_mask, ripe_mask, overripe_mask, avg_sat, green_pct, dark_pct

    def calculate_texture_variance(self, hsv_img, fruit_mask):
        gray = cv2.cvtColor(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        variance = np.clip(variance, 0, None)

        fruit_pixels = fruit_mask > 0
        avg_variance = np.mean(variance[fruit_pixels]) if np.sum(fruit_pixels) > 0 else 0

        return avg_variance, variance

    def refine_with_texture(self, hsv_img, unripe_mask, ripe_mask, overripe_mask, fruit_mask, avg_variance, green_pct, avg_sat, dark_pct):
        if green_pct > 35:
            return unripe_mask, ripe_mask, overripe_mask, 0

        if avg_sat > 180 and dark_pct < 3:
            return unripe_mask, ripe_mask, overripe_mask, 0

        gray = cv2.cvtColor(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance_map = sqr_mean - mean**2
        variance_map = np.clip(variance_map, 0, None)

        if np.max(variance_map) > 0:
            variance_norm = (variance_map / np.max(variance_map) * 255).astype(np.uint8)
        else:
            variance_norm = variance_map.astype(np.uint8)

        _, high_variance_mask = cv2.threshold(variance_norm, 35, 255, cv2.THRESH_BINARY)
        high_variance_mask = cv2.bitwise_and(high_variance_mask, fruit_mask)

        _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_mask, fruit_mask)

        texture_overripe = cv2.bitwise_or(high_variance_mask, dark_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        texture_overripe = cv2.morphologyEx(texture_overripe, cv2.MORPH_OPEN, kernel, iterations=1)

        texture_overripe_pct = np.sum(texture_overripe > 0) / np.sum(fruit_mask > 0) * 100 if np.sum(fruit_mask > 0) > 0 else 0

        if avg_variance > 2000 and texture_overripe_pct > 15 and dark_pct > 5:
            overripe_additional = cv2.bitwise_and(texture_overripe, ripe_mask)
            overripe_mask = cv2.bitwise_or(overripe_mask, overripe_additional)
            ripe_mask = cv2.bitwise_and(ripe_mask, cv2.bitwise_not(overripe_additional))

            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            overripe_expanded = cv2.dilate(overripe_mask, kernel_dilate, iterations=1)
            overripe_from_ripe = cv2.bitwise_and(overripe_expanded, ripe_mask)
            overripe_mask = cv2.bitwise_or(overripe_mask, overripe_from_ripe)
            ripe_mask = cv2.bitwise_and(ripe_mask, cv2.bitwise_not(overripe_from_ripe))

        return unripe_mask, ripe_mask, overripe_mask, texture_overripe_pct

    def create_visualization(self, bgr_img, fruit_mask, unripe_mask, ripe_mask, overripe_mask):
        overlay = bgr_img.copy()

        overlay[unripe_mask > 0] = [0, 255, 0]
        overlay[ripe_mask > 0] = [0, 255, 255]
        overlay[overripe_mask > 0] = [0, 0, 255]

        result = cv2.addWeighted(bgr_img, 0.6, overlay, 0.4, 0)

        contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 2)

        return result

    def process_fruit(self, image_path):
        img = self.load_image(image_path)
        hsv = self.preprocess_image(img)

        self.fruit_mask = self.segment_fruit_from_background(hsv)
        edges = self.detect_edges(hsv)

        unripe_mask, ripe_mask, overripe_mask, avg_sat, green_pct, dark_pct = self.segment_ripeness_by_color(hsv, self.fruit_mask)

        avg_variance, variance_map = self.calculate_texture_variance(hsv, self.fruit_mask)

        self.unripe_mask, self.ripe_mask, self.overripe_mask, texture_overripe_pct = \
            self.refine_with_texture(hsv, unripe_mask, ripe_mask, overripe_mask, self.fruit_mask, avg_variance, green_pct, avg_sat, dark_pct)

        unripe_area = np.sum(self.unripe_mask > 0)
        ripe_area = np.sum(self.ripe_mask > 0)
        overripe_area = np.sum(self.overripe_mask > 0)

        total_area = unripe_area + ripe_area + overripe_area
        if total_area == 0:
            unripe_pct = ripe_pct = overripe_pct = 0
        else:
            unripe_pct = (unripe_area / total_area) * 100
            ripe_pct = (ripe_area / total_area) * 100
            overripe_pct = (overripe_area / total_area) * 100

        if green_pct > 40 or unripe_pct > 50:
            classification = "Unripe"
        elif dark_pct > 8 or (overripe_pct > 15 and dark_pct > 5):
            classification = "Overripe"
        elif avg_sat > 180 and ripe_pct > 45 and dark_pct < 3:
            classification = "Ripe"
        elif avg_variance > 2500 and green_pct < 20 and overripe_pct > 10 and dark_pct > 5:
            classification = "Overripe"
        elif texture_overripe_pct > 25 and dark_pct > 5:
            classification = "Overripe"
        elif avg_sat < 60 and unripe_pct > 35:
            classification = "Unripe"
        elif ripe_pct > 50:
            classification = "Ripe"
        else:
            max_pct = max(unripe_pct, ripe_pct, overripe_pct)
            if max_pct == unripe_pct:
                classification = "Unripe"
            elif max_pct == ripe_pct:
                classification = "Ripe"
            else:
                classification = "Overripe"

        bgr_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result_img = self.create_visualization(bgr_img, self.fruit_mask, self.unripe_mask,
                                                self.ripe_mask, self.overripe_mask)

        return {
            'original': img,
            'hsv': hsv,
            'edges': edges,
            'fruit_mask': self.fruit_mask,
            'unripe_mask': self.unripe_mask,
            'ripe_mask': self.ripe_mask,
            'overripe_mask': self.overripe_mask,
            'result': result_img,
            'variance_map': variance_map,
            'unripe_pct': unripe_pct,
            'ripe_pct': ripe_pct,
            'overripe_pct': overripe_pct,
            'classification': classification,
            'avg_variance': avg_variance,
            'texture_overripe_pct': texture_overripe_pct,
            'avg_sat': avg_sat,
            'green_pct': green_pct,
            'dark_pct': dark_pct
        }


class FruitRipenessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Ripeness Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        self.detector = FruitRipenessDetector()
        self.current_image_path = None
        self.results = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text=" Fruit Ripeness Detection System ", 
            font=('Helvetica', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(expand=True)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Controls and Results
        left_panel = tk.Frame(main_container, bg='white', width=350, relief=tk.RIDGE, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5), pady=0)
        left_panel.pack_propagate(False)
        
        # Upload Button
        button_frame = tk.Frame(left_panel, bg='white')
        button_frame.pack(fill=tk.X, padx=15, pady=20)
        
        self.upload_btn = tk.Button(
            button_frame,
            text=" Upload Fruit Image",
            command=self.upload_image,
            font=('Helvetica', 12, 'bold'),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            cursor='hand2',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=15
        )
        self.upload_btn.pack(fill=tk.X)
        
        # Process Button
        self.process_btn = tk.Button(
            button_frame,
            text="🔍 Analyze Ripeness",
            command=self.process_image,
            font=('Helvetica', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            cursor='hand2',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=15,
            state=tk.DISABLED
        )
        self.process_btn.pack(fill=tk.X, pady=(10, 0))
        
        # Results Section
        results_frame = tk.LabelFrame(
            left_panel,
            text="Analysis Results",
            font=('Helvetica', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief=tk.GROOVE,
            bd=2
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Classification Result
        self.classification_frame = tk.Frame(results_frame, bg='white')
        self.classification_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            self.classification_frame,
            text="Overall Classification:",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            fg='#34495e'
        ).pack(anchor='w')
        
        self.classification_label = tk.Label(
            self.classification_frame,
            text="No analysis yet",
            font=('Helvetica', 18, 'bold'),
            bg='white',
            fg='#7f8c8d',
            pady=10
        )
        self.classification_label.pack()
        
        # Ripeness Percentages
        percentages_frame = tk.LabelFrame(
            results_frame,
            text="Ripeness Distribution",
            font=('Helvetica', 10, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        percentages_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.unripe_label = self.create_metric_label(percentages_frame, "Unripe:", "0.00%", '#27ae60')
        self.ripe_label = self.create_metric_label(percentages_frame, "Ripe:", "0.00%", '#f39c12')
        self.overripe_label = self.create_metric_label(percentages_frame, "Overripe:", "0.00%", '#e74c3c')
        
        # Metrics
        metrics_frame = tk.LabelFrame(
            results_frame,
            text="Detailed Metrics",
            font=('Helvetica', 10, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        metrics_scroll = tk.Scrollbar(metrics_frame)
        metrics_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.metrics_text = tk.Text(
            metrics_frame,
            height=12,
            font=('Courier', 9),
            bg='#ecf0f1',
            fg='#2c3e50',
            relief=tk.FLAT,
            padx=10,
            pady=10,
            yscrollcommand=metrics_scroll.set
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        metrics_scroll.config(command=self.metrics_text.yview)
        self.metrics_text.insert(tk.END, "Upload and analyze an image to see detailed metrics...")
        self.metrics_text.config(state=tk.DISABLED)
        
        # Right panel - Image Display
        right_panel = tk.Frame(main_container, bg='white', relief=tk.RIDGE, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=0)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_tabs()
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready to analyze fruit images",
            font=('Helvetica', 9),
            bg='#34495e',
            fg='white',
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, padx=10)
        
    def create_metric_label(self, parent, label_text, value_text, color):
        frame = tk.Frame(parent, bg='white')
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            frame,
            text=label_text,
            font=('Helvetica', 10),
            bg='white',
            fg='#34495e'
        ).pack(side=tk.LEFT)
        
        value_label = tk.Label(
            frame,
            text=value_text,
            font=('Helvetica', 10, 'bold'),
            bg='white',
            fg=color
        )
        value_label.pack(side=tk.RIGHT)
        
        return value_label
    
    def create_tabs(self):
        # Tab 1: Original & Result
        self.tab1 = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab1, text='Original & Result')
        
        # Tab 2: Processing Steps
        self.tab2 = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab2, text='Processing Steps')
        
        # Tab 3: Ripeness Masks
        self.tab3 = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab3, text='Ripeness Masks')
        
        # Tab 4: Color Segmentation Steps
        self.tab4 = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab4, text='Color Segmentation')
        
        # Tab 5: Texture Analysis
        self.tab5 = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab5, text='Texture Analysis')
        
        # Create image labels for each tab
        self.create_image_display(self.tab1, "original_result")
        self.create_image_display(self.tab2, "processing")
        self.create_image_display(self.tab3, "masks")
        self.create_image_display(self.tab4, "color_segmentation")
        self.create_image_display(self.tab5, "texture")
        
    def create_image_display(self, parent, tab_type):
        canvas_frame = tk.Frame(parent, bg='white')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        if tab_type == "original_result":
            # Two images side by side
            left_frame = tk.Frame(canvas_frame, bg='white')
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            tk.Label(left_frame, text="Original Image", font=('Helvetica', 11, 'bold'), bg='white').pack()
            self.original_canvas = tk.Label(left_frame, bg='#ecf0f1', text="No image loaded", 
                                           font=('Helvetica', 12), fg='#95a5a6')
            self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            right_frame = tk.Frame(canvas_frame, bg='white')
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
            
            tk.Label(right_frame, text="Analysis Result", font=('Helvetica', 11, 'bold'), bg='white').pack()
            self.result_canvas = tk.Label(right_frame, bg='#ecf0f1', text="Analysis not performed", 
                                         font=('Helvetica', 12), fg='#95a5a6')
            self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
        elif tab_type == "processing":
            self.processing_canvas = tk.Label(canvas_frame, bg='#ecf0f1')
            self.processing_canvas.pack(fill=tk.BOTH, expand=True)
            
        elif tab_type == "masks":
            self.masks_canvas = tk.Label(canvas_frame, bg='#ecf0f1')
            self.masks_canvas.pack(fill=tk.BOTH, expand=True)
            
        elif tab_type == "color_segmentation":
            self.color_segmentation_canvas = tk.Label(canvas_frame, bg='#ecf0f1')
            self.color_segmentation_canvas.pack(fill=tk.BOTH, expand=True)
            
        elif tab_type == "texture":
            self.texture_canvas = tk.Label(canvas_frame, bg='#ecf0f1')
            self.texture_canvas.pack(fill=tk.BOTH, expand=True)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select a fruit image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_original_image(file_path)
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
            
            # Clear previous results
            self.classification_label.config(text="Ready to analyze", fg='#3498db')
            self.unripe_label.config(text="0.00%")
            self.ripe_label.config(text="0.00%")
            self.overripe_label.config(text="0.00%")
            
    def display_original_image(self, image_path):
        img = Image.open(image_path)
        img = self.resize_image(img, (450, 450))
        photo = ImageTk.PhotoImage(img)
        
        self.original_canvas.config(image=photo, text="")
        self.original_canvas.image = photo
        
    def resize_image(self, img, max_size):
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img
    
    def process_image(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        
        self.status_label.config(text="Processing... Please wait")
        self.process_btn.config(state=tk.DISABLED, text="Processing...")
        self.root.update()
        
        # Run processing in a thread to keep UI responsive
        thread = threading.Thread(target=self.run_detection)
        thread.start()
        
    def run_detection(self):
        try:
            self.results = self.detector.process_fruit(self.current_image_path)
            self.root.after(0, self.display_results)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL, text="🔍 Analyze Ripeness"))
            self.root.after(0, lambda: self.status_label.config(text="Error during processing"))
    
    def display_results(self):
        if not self.results:
            return
        
        # Update classification
        classification = self.results['classification']
        color_map = {
            'Unripe': '#27ae60',
            'Ripe': '#f39c12',
            'Overripe': '#e74c3c'
        }
        
        self.classification_label.config(
            text=classification,
            fg=color_map.get(classification, '#7f8c8d')
        )
        
        # Update percentages
        self.unripe_label.config(text=f"{self.results['unripe_pct']:.2f}%")
        self.ripe_label.config(text=f"{self.results['ripe_pct']:.2f}%")
        self.overripe_label.config(text=f"{self.results['overripe_pct']:.2f}%")
        
        # Update metrics
        self.update_metrics_text()
        
        # Display result image
        self.display_result_image()
        
        # Display processing steps
        self.display_processing_steps()
        
        # Display masks
        self.display_masks()
        
        # Display color segmentation steps
        self.display_color_segmentation()
        
        # Display texture analysis
        self.display_texture_analysis()
        
        self.process_btn.config(state=tk.NORMAL, text="🔍 Analyze Ripeness")
        self.status_label.config(text=f"Analysis complete: {classification}")
        
    def update_metrics_text(self):
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        
        metrics_text = f"""
╔══════════════════════════════════╗
║      TEXTURE ANALYSIS            ║
╚══════════════════════════════════╝
  Average Variance:    {self.results['avg_variance']:.2f}
  Texture Overripe %:  {self.results['texture_overripe_pct']:.2f}%

╔══════════════════════════════════╗
║       COLOR METRICS              ║
╚══════════════════════════════════╝
  Average Saturation:  {self.results['avg_sat']:.2f}
  Green Percentage:    {self.results['green_pct']:.2f}%
  Dark Pixel %:        {self.results['dark_pct']:.2f}%

╔══════════════════════════════════╗
║    RIPENESS DISTRIBUTION         ║
╚══════════════════════════════════╝
  Unripe (Green):      {self.results['unripe_pct']:.2f}%
  Ripe (Yellow):       {self.results['ripe_pct']:.2f}%
  Overripe (Brown):    {self.results['overripe_pct']:.2f}%
"""
        
        self.metrics_text.insert(tk.END, metrics_text)
        self.metrics_text.config(state=tk.DISABLED)
        
    def display_result_image(self):
        result_bgr = self.results['result']
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(result_rgb)
        img = self.resize_image(img, (450, 450))
        photo = ImageTk.PhotoImage(img)
        
        self.result_canvas.config(image=photo, text="")
        self.result_canvas.image = photo
        
    def display_processing_steps(self):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.patch.set_facecolor('white')
        
        # Row 1: Original and Color Space Conversions
        original_rgb = cv2.cvtColor(self.results['original'], cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('1. Original Image', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        
        # HSV conversion
        hsv_rgb = cv2.cvtColor(self.results['hsv'], cv2.COLOR_HSV2RGB)
        axes[0, 1].imshow(hsv_rgb)
        axes[0, 1].set_title('2. HSV Color Space', fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')
        
        # After Gaussian Blur
        axes[0, 2].imshow(hsv_rgb)
        axes[0, 2].set_title('3. After Gaussian Blur', fontsize=10, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: HSV Channels
        h, s, v = cv2.split(self.results['hsv'])
        
        axes[1, 0].imshow(h, cmap='hsv')
        axes[1, 0].set_title('4. Hue Channel', fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(s, cmap='gray')
        axes[1, 1].set_title('5. Saturation Channel', fontsize=10, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(v, cmap='gray')
        axes[1, 2].set_title('6. Value Channel', fontsize=10, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Row 3: Edge Detection and Segmentation
        axes[2, 0].imshow(self.results['edges'], cmap='gray')
        axes[2, 0].set_title('7. Edge Detection (Canny)', fontsize=10, fontweight='bold')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(self.results['fruit_mask'], cmap='gray')
        axes[2, 1].set_title('8. Fruit Segmentation Mask', fontsize=10, fontweight='bold')
        axes[2, 1].axis('off')
        
        # Masked fruit
        masked_fruit = cv2.bitwise_and(original_rgb, original_rgb, mask=self.results['fruit_mask'])
        axes[2, 2].imshow(masked_fruit)
        axes[2, 2].set_title('9. Segmented Fruit', fontsize=10, fontweight='bold')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        self.embed_matplotlib_figure(fig, self.processing_canvas)
        
    def display_masks(self):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.patch.set_facecolor('white')
        
        # Unripe
        axes[0, 0].imshow(self.results['unripe_mask'], cmap='Greens')
        axes[0, 0].set_title(f"Unripe Mask ({self.results['unripe_pct']:.1f}%)", 
                            fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ripe
        axes[0, 1].imshow(self.results['ripe_mask'], cmap='YlOrBr')
        axes[0, 1].set_title(f"Ripe Mask ({self.results['ripe_pct']:.1f}%)", 
                            fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overripe
        axes[1, 0].imshow(self.results['overripe_mask'], cmap='Reds')
        axes[1, 0].set_title(f"Overripe Mask ({self.results['overripe_pct']:.1f}%)", 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Result
        result_rgb = cv2.cvtColor(self.results['result'], cv2.COLOR_BGR2RGB)
        axes[1, 1].imshow(result_rgb)
        axes[1, 1].set_title(f"Result: {self.results['classification']}", 
                            fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        self.embed_matplotlib_figure(fig, self.masks_canvas)
        
    def display_color_segmentation(self):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.patch.set_facecolor('white')
        
        # Get HSV channels
        h, s, v = cv2.split(self.results['hsv'])
        fruit_mask = self.results['fruit_mask']
        
        # Row 1: Masked HSV channels
        h_masked = cv2.bitwise_and(h, h, mask=fruit_mask)
        s_masked = cv2.bitwise_and(s, s, mask=fruit_mask)
        v_masked = cv2.bitwise_and(v, v, mask=fruit_mask)
        
        axes[0, 0].imshow(h_masked, cmap='hsv')
        axes[0, 0].set_title('1. Masked Hue Channel', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(s_masked, cmap='gray')
        axes[0, 1].set_title('2. Masked Saturation', fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(v_masked, cmap='gray')
        axes[0, 2].set_title('3. Masked Value', fontsize=10, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Color range detection for unripe (green)
        green_range = cv2.inRange(h_masked, 40, 85)
        green_range = cv2.bitwise_and(green_range, fruit_mask)
        
        axes[1, 0].imshow(green_range, cmap='Greens')
        axes[1, 0].set_title('4. Green Detection (40-85° Hue)', fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Color range for ripe (red/yellow)
        red_range1 = cv2.inRange(h_masked, 0, 10)
        red_range2 = cv2.inRange(h_masked, 170, 180)
        red_range = cv2.bitwise_or(red_range1, red_range2)
        red_range = cv2.bitwise_and(red_range, fruit_mask)
        
        axes[1, 1].imshow(red_range, cmap='Reds')
        axes[1, 1].set_title('5. Red Detection (0-10°, 170-180°)', fontsize=10, fontweight='bold')
        axes[1, 1].axis('off')
        
        yellow_range = cv2.inRange(h_masked, 15, 50)
        yellow_range = cv2.bitwise_and(yellow_range, fruit_mask)
        
        axes[1, 2].imshow(yellow_range, cmap='YlOrBr')
        axes[1, 2].set_title('6. Yellow Detection (15-50°)', fontsize=10, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Row 3: Dark pixels and final masks
        dark_pixels = cv2.inRange(v_masked, 0, 100)
        dark_pixels = cv2.bitwise_and(dark_pixels, fruit_mask)
        
        axes[2, 0].imshow(dark_pixels, cmap='gray')
        axes[2, 0].set_title(f'7. Dark Pixels ({self.results["dark_pct"]:.1f}%)', fontsize=10, fontweight='bold')
        axes[2, 0].axis('off')
        
        # Combined unripe mask
        axes[2, 1].imshow(self.results['unripe_mask'], cmap='Greens')
        axes[2, 1].set_title(f'8. Final Unripe Mask', fontsize=10, fontweight='bold')
        axes[2, 1].axis('off')
        
        # Combined overripe mask
        axes[2, 2].imshow(self.results['overripe_mask'], cmap='Reds')
        axes[2, 2].set_title(f'9. Final Overripe Mask', fontsize=10, fontweight='bold')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        self.embed_matplotlib_figure(fig, self.color_segmentation_canvas)
        
    def display_texture_analysis(self):
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.patch.set_facecolor('white')
        
        # Get grayscale image
        gray = cv2.cvtColor(cv2.cvtColor(self.results['hsv'], cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        fruit_mask = self.results['fruit_mask']
        
        # Row 1: Grayscale and Variance
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title('1. Grayscale Image', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Variance Map
        variance_map = self.results['variance_map']
        if np.max(variance_map) > 0:
            variance_viz = (variance_map / np.max(variance_map) * 255).astype(np.uint8)
        else:
            variance_viz = variance_map.astype(np.uint8)
        
        im1 = axes[0, 1].imshow(variance_viz, cmap='hot')
        axes[0, 1].set_title(f"2. Texture Variance Map", fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # High variance areas
        _, high_variance_mask = cv2.threshold(variance_viz, 35, 255, cv2.THRESH_BINARY)
        high_variance_mask = cv2.bitwise_and(high_variance_mask, fruit_mask)
        
        axes[0, 2].imshow(high_variance_mask, cmap='hot')
        axes[0, 2].set_title('3. High Variance Regions', fontsize=10, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Dark regions and Final result
        _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_mask, fruit_mask)
        
        axes[1, 0].imshow(dark_mask, cmap='gray')
        axes[1, 0].set_title(f'4. Dark Regions (V < 100)', fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Pie Chart
        sizes = [self.results['unripe_pct'], self.results['ripe_pct'], self.results['overripe_pct']]
        labels = ['Unripe', 'Ripe', 'Overripe']
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        explode = (0.05, 0.05, 0.05)
        
        axes[1, 1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[1, 1].set_title('5. Ripeness Distribution', fontsize=10, fontweight='bold')
        
        # Final classification result
        result_rgb = cv2.cvtColor(self.results['result'], cv2.COLOR_BGR2RGB)
        axes[1, 2].imshow(result_rgb)
        axes[1, 2].set_title(f"6. Final: {self.results['classification']}", fontsize=10, fontweight='bold', 
                           color=self.get_classification_color(self.results['classification']))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        self.embed_matplotlib_figure(fig, self.texture_canvas)
    
    def get_classification_color(self, classification):
        color_map = {
            'Unripe': '#27ae60',
            'Ripe': '#f39c12',
            'Overripe': '#e74c3c'
        }
        return color_map.get(classification, '#7f8c8d')
        
    def embed_matplotlib_figure(self, fig, canvas_widget):
        # Convert matplotlib figure to image
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        
        # Convert RGBA to RGB
        img_rgb = img[:, :, :3]
        
        # Convert to PIL and display
        pil_img = Image.fromarray(img_rgb)
        pil_img = self.resize_image(pil_img, (1000, 600))
        photo = ImageTk.PhotoImage(pil_img)
        
        canvas_widget.config(image=photo)
        canvas_widget.image = photo
        
        plt.close(fig)


def main():
    root = tk.Tk()
    app = FruitRipenessGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
