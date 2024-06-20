import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import json


COLORS = [
    'red',
    'red2',
    'red3',
    'red4',
    'DeepPink2',
    'DeepPink3',
]


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer with Points")

        self.image = None
        self.photo = None

        # Initialize directory and file variables
        self.current_directory = tk.StringVar()
        self.current_file = tk.StringVar()

        # Dropdown to select directory
        self.dir_label = tk.Label(root, text="Select Directory:")
        self.dir_label.grid(row=0, column=0, padx=10, pady=5)

        self.directories = [d for d in os.listdir('.') if (os.path.isdir(d) and
                                                           not any(i in d for i in ['py', 'json', 'video']))]
        self.dir_dropdown = ttk.Combobox(root, textvariable=self.current_directory, values=self.directories)
        self.dir_dropdown.grid(row=0, column=1, padx=10, pady=5)
        self.dir_dropdown.bind("<<ComboboxSelected>>", self.update_file_list)

        # Dropdown to select file
        self.file_label = tk.Label(root, text="Select File:")
        self.file_label.grid(row=1, column=0, padx=10, pady=5)

        self.files = []
        self.file_dropdown = ttk.Combobox(root, textvariable=self.current_file, values=self.files)
        self.file_dropdown.grid(row=1, column=1, padx=10, pady=5)
        self.file_dropdown.bind("<<ComboboxSelected>>", self.load_image)

        # Scrollable canvas to display image
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.grid(row=0, column=0, sticky='nsew')

        # Scrollbars for the canvas
        self.vsb = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.vsb.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.hsb = tk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.hsb.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(xscrollcommand=self.hsb.set)

        # Frame to hold the canvas and scrollbars
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        # Bind the canvas resizing to image resizing
        self.root.bind("<Configure>", self.adjust_canvas_size)

        # Bind mouse click event to the canvas to log coordinates
        self.canvas.bind("<Button-1>", self.log_click_coordinates)

    def update_file_list(self, event=None):
        directory = self.current_directory.get()
        self.files = sorted(f for f in os.listdir(directory) if f.endswith('.png'))
        self.file_dropdown['values'] = self.files

        if self.files:
            self.current_file.set(self.files[0])
            self.load_image()

    def load_image(self, event=None):
        directory = self.current_directory.get()
        filename = self.current_file.get()
        path = os.path.join(directory, filename)

        self.image = Image.open(path)
        self.update_canvas(self.image)

    def update_canvas(self, image):
        self.photo = ImageTk.PhotoImage(image)

        # Adjust canvas size to image size
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.adjust_canvas_size(None)
        self.load_points()

    def load_points(self):
        directory = self.current_directory.get()
        filename = self.current_file.get()
        base, ext = os.path.splitext(filename)
        json_directory = f"{directory}_json"
        json_file = f"{base}.json"

        json_path = os.path.join(json_directory, json_file)

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if data['curvature']['top']['x'] is None:
                        data['curvature']['top']['x'] = (data['top']['left']['x'] + data['top']['right']['x']) / 2
                    if data['curvature']['top']['y'] is None:
                        data['curvature']['top']['y'] = (data['top']['left']['y'] + data['top']['right']['y']) / 2
                    if data['curvature']['bottom']['x'] is None:
                        data['curvature']['bottom']['x'] = (data['bottom']['left']['x'] + data['bottom']['right'][
                            'x']) / 2
                    if data['curvature']['bottom']['y'] is None:
                        data['curvature']['bottom']['y'] = (data['bottom']['left']['y'] + data['bottom']['right'][
                            'y']) / 2
                    points = [
                        data['top']['left'],
                        data['top']['right'],
                        data['bottom']['left'],
                        data['bottom']['right'],
                        data['curvature']['top'],
                        data['curvature']['bottom'],
                    ]
            except Exception as e:
                print(f"Error loading JSON file: {e}")
                points = []

            # Draw points on the image
            for point, color in zip(points, COLORS):
                x, y = point['x'], point['y']
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color)

    def adjust_canvas_size(self, event):
        # Check if there is an image loaded
        if self.image:
            image_width, image_height = self.image.width, self.image.height
            # Set scroll region to accommodate the full image
            self.canvas.configure(scrollregion=(0, 0, image_width, image_height))

            # Adjust the size of the canvas to the lesser of the image size or the window size
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()

            # Calculate the usable size minus padding and scrollbar thickness
            usable_width = window_width - 40
            usable_height = window_height - 150

            canvas_width = min(usable_width, image_width)
            canvas_height = min(usable_height, image_height)

            self.canvas.config(width=canvas_width, height=canvas_height)

    def log_click_coordinates(self, event):
        # Get the canvas x and y positions
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Check if the click is within the bounds of the image
        if (0 <= canvas_x < self.photo.width()) and (0 <= canvas_y < self.photo.height()):
            # Get the image x and y positions
            image_x = int(self.image.width * (canvas_x / self.canvas.winfo_width()))
            image_y = int(self.image.height * (canvas_y / self.canvas.winfo_height()))
            print(f"Clicked at image coordinates: ({image_x}, {image_y})")
        else:
            print("Clicked outside the bounds of the image.")


if __name__ == "__main__":
    _root = tk.Tk()
    app = ImageApp(_root)
    _root.mainloop()
