import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── General Parameters (Fixed Global Constants) ───────────────────────────────
HARRIS_BLOCK_SIZE = 3
HARRIS_KSIZE      = 3
SCALE_SIGMA_BASE  = 1.0
DOG_SIGMA1        = 1.0
DOG_SIGMA2        = 5.0

# ── MATHEMATICAL IMPLEMENTATION (Strictly Intact) ─────────────────────────────

def preprocess(gray: np.ndarray, is_low_res: bool = False) -> np.ndarray:
    u8 = gray.astype(np.uint8)
    d, sc, ss = (11, 80, 80) if is_low_res else (5, 40, 40)
    bil   = cv2.bilateralFilter(u8, d=d, sigmaColor=sc, sigmaSpace=ss)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(bil).astype(np.float32)

def simulate_low_res(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    h, w    = img_bgr.shape[:2]
    small   = cv2.resize(img_bgr,
                         (max(1, int(w * factor)), max(1, int(h * factor))),
                         interpolation=cv2.INTER_AREA)
    up      = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    sigma   = (1.0 / factor) * 0.6
    return cv2.GaussianBlur(up, (0, 0), sigmaX=sigma, sigmaY=sigma)

def _nms(R: np.ndarray, candidates: np.ndarray, radius: int) -> np.ndarray:
    if len(candidates) == 0:
        return candidates
    scores     = R[candidates[:, 0], candidates[:, 1]]
    order      = np.argsort(-scores)
    candidates = candidates[order]
    suppressed = np.zeros(len(candidates), dtype=bool)
    kept = []
    for i in range(len(candidates)):
        if suppressed[i]:
            continue
        kept.append(i)
        r0, c0 = candidates[i]
        dists        = np.maximum(np.abs(candidates[:, 0] - r0),
                                  np.abs(candidates[:, 1] - c0))
        suppressed  |= dists < radius
        suppressed[i] = False
    return candidates[kept]

def draw_corners(img_bgr: np.ndarray, corners_xy, color, size: int = 6,
                 thickness: int = 1) -> np.ndarray:
    out = img_bgr.copy()
    for x, y in corners_xy:
        cx, cy = int(x), int(y)
        cv2.line(out, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
        cv2.line(out, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 2, color, -1, cv2.LINE_AA)
    return out

def standard_harris(gray_f32: np.ndarray, block_size: int, ksize: int,
                    k: float, threshold_ratio: float,
                    nms_r: int) -> np.ndarray:
    R = cv2.cornerHarris(gray_f32, block_size, ksize, k)
    R = cv2.dilate(R, None)
    thresh     = threshold_ratio * R.max()
    candidates = np.argwhere(R > thresh)
    corners = _nms(R, candidates, nms_r)
    return corners[:, ::-1]

def _morphological_enhance(gray_u8: np.ndarray) -> np.ndarray:
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grad = cv2.dilate(gray_u8, k) - cv2.erode(gray_u8, k)
    return cv2.addWeighted(gray_u8, 0.65, grad, 0.35, 0)

def _dog_structural_mask(gray_u8: np.ndarray, sigma1: float, sigma2: float,
                          thresh_ratio: float) -> np.ndarray:
    f   = gray_u8.astype(np.float32)
    g1  = cv2.GaussianBlur(f, (0, 0), sigma1)
    g2  = cv2.GaussianBlur(f, (0, 0), sigma2)
    dog = np.abs(g2 - g1)
    dog /= (dog.max() + 1e-8)
    binary = (dog > thresh_ratio).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(binary, kernel)

def scale_invariant_harris_improved(gray_f32: np.ndarray, num_scales: int,
                                     sigma_base: float, block_size: int,
                                     k: float, threshold_ratio: float,
                                     nms_r: int,
                                     dog_sigma1: float = DOG_SIGMA1,
                                     dog_sigma2: float = DOG_SIGMA2,
                                     dog_thresh: float = 0.07
                                     ) -> np.ndarray:
    gray_u8 = np.clip(gray_f32, 0, 255).astype(np.uint8)
    enhanced = _morphological_enhance(gray_u8).astype(np.float32)
    dog_mask = _dog_structural_mask(gray_u8, dog_sigma1, dog_sigma2, dog_thresh)

    scale_responses = []
    for s in range(num_scales):
        sigma   = sigma_base * (2 ** s)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=sigma, sigmaY=sigma)
        R       = cv2.cornerHarris(blurred, block_size, 3, k)
        R_norm  = R * (sigma ** 2)
        scale_responses.append(R_norm)

    stack = np.stack(scale_responses, axis=0)
    R_max = stack.max(axis=0)
    R_max[dog_mask == 0] = 0

    thresh     = threshold_ratio * R_max.max() if R_max.max() > 0 else 1
    candidates = np.argwhere(R_max > thresh)
    corners    = _nms(R_max, candidates, nms_r)
    return corners[:, ::-1]


# ── GUI APPLICATION ───────────────────────────────────────────────────────────

class HarrisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Harris Corner Detection UI")
        self.geometry("1200x800")
        
        # State variables
        self.img_bgr = None
        self.is_dark_mode = False
        
        # Theme definitions
        self.themes = {
            "light": {"bg": "#f0f0f0", "fg": "#000000", "entry_bg": "#ffffff", "entry_fg": "#000000", "plot_bg": "#ffffff"},
            "dark":  {"bg": "#2b2b2b", "fg": "#ffffff", "entry_bg": "#404040", "entry_fg": "#ffffff", "plot_bg": "#3c3f41"}
        }

        # 1. Left Panel (Controls)
        self.control_frame = tk.Frame(self, width=320)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)

        # Theme Toggle
        self.btn_theme = tk.Button(self.control_frame, text="🌙 Toggle Dark Mode", command=self.toggle_theme, font=("Arial", 9))
        self.btn_theme.pack(fill=tk.X, pady=(0, 10))

        # Upload Button
        self.btn_upload = tk.Button(self.control_frame, text="📁 Upload Image", command=self.upload_image, font=("Arial", 10, "bold"), bg="#1a73e8", fg="white")
        self.btn_upload.pack(fill=tk.X, pady=5)

        self.lbl_file = tk.Label(self.control_frame, text="No image selected")
        self.lbl_file.pack(fill=tk.X, pady=5)

        tk.Frame(self.control_frame, height=2, bg="#ccc").pack(fill=tk.X, pady=5)
        self.lbl_params = tk.Label(self.control_frame, text="⚙ Parameters", font=("Arial", 11, "bold"))
        self.lbl_params.pack(anchor="w")

        # Dictionary for text input variables
        self.params = {
            "harris_k": tk.DoubleVar(value=0.05),
            "hi_thresh": tk.DoubleVar(value=0.015),
            "lo_thresh": tk.DoubleVar(value=0.05),
            "si_hi_thresh": tk.DoubleVar(value=0.025),
            "si_lo_thresh": tk.DoubleVar(value=0.06),
            "dog_thresh": tk.DoubleVar(value=0.07),
            "scale_factor": tk.DoubleVar(value=0.25),
            "nms_radius": tk.IntVar(value=12),
            "num_scales": tk.IntVar(value=5)
        }

        labels = {
            "harris_k": "Harris k (Sensitivity):",
            "hi_thresh": "Std Hi-Res Threshold:",
            "lo_thresh": "Std Lo-Res Threshold:",
            "si_hi_thresh": "SI Hi-Res Threshold:",
            "si_lo_thresh": "SI Lo-Res Threshold:",
            "dog_thresh": "DoG Mask Threshold:",
            "scale_factor": "Lo-Res Scale Factor:",
            "nms_radius": "NMS Radius (px):",
            "num_scales": "Number of Scales (SI):"
        }

        # Create text input boxes and store references for theme toggling
        self.entry_widgets = []
        self.label_widgets = [self.lbl_file, self.lbl_params]
        
        for key, var in self.params.items():
            frame = tk.Frame(self.control_frame)
            frame.pack(fill=tk.X, pady=2)
            lbl = tk.Label(frame, text=labels[key])
            lbl.pack(side=tk.LEFT)
            ent = tk.Entry(frame, textvariable=var, width=8)
            ent.pack(side=tk.RIGHT)
            self.label_widgets.append(lbl)
            self.entry_widgets.append(ent)
            self.entry_widgets.append(frame) # add frame to update its bg

        tk.Frame(self.control_frame, height=2, bg="#ccc").pack(fill=tk.X, pady=10)

        # Run Button
        self.btn_run = tk.Button(self.control_frame, text="▶ Run Detection", command=self.run_detection, font=("Arial", 10, "bold"), bg="#28a745", fg="white")
        self.btn_run.pack(fill=tk.X, pady=5)
        
        self.lbl_status = tk.Label(self.control_frame, text="Status: Ready", fg="green")
        self.lbl_status.pack(fill=tk.X, pady=5)
        self.label_widgets.append(self.lbl_status)

        # ─── Parameter Guide (Directions & Symbols) ───
        guide_text = (
            "📖 PARAMETER GUIDE\n\n"
            "• Harris k (Sensitivity): Identifies corners.\n"
            "  ↑ Increase: Fewer edges reported as corners.\n"
            "  ↓ Decrease: More corners, but risk of false positives.\n\n"
            "• Thresholds (Std & SI): Minimum corner strength.\n"
            "  ↑ Increase: Stricter. Keeps only the sharpest corners.\n"
            "  ↓ Decrease: Lenient. Detects faint, low-contrast corners.\n\n"
            "• DoG Mask Threshold: Filters out vegetation/texture.\n"
            "  ↑ Increase: Stricter filtering of natural textures.\n"
            "  ↓ Decrease: Allows more corners in textured areas.\n\n"
            "• Lo-Res Scale Factor: Simulates satellite quality.\n"
            "  ↑ Increase (e.g. 0.8): Minimal blurring/downsampling.\n"
            "  ↓ Decrease (e.g. 0.2): Aggressive blurring (worse sensor).\n\n"
            "• NMS Radius (px): Non-Maximum Suppression.\n"
            "  ↑ Increase: Corners are forced to be further apart.\n"
            "  ↓ Decrease: Allows dense clustering of corners.\n\n"
            "• Number of Scales (SI): Multi-scale feature search.\n"
            "  ↑ Increase: Better scale-invariance, but runs slower."
        )
        self.txt_guide = tk.Text(self.control_frame, wrap=tk.WORD, height=14, font=("Consolas", 8))
        self.txt_guide.insert(tk.END, guide_text)
        self.txt_guide.config(state=tk.DISABLED)
        self.txt_guide.pack(fill=tk.BOTH, expand=True, pady=10)

        # 2. Right Panel (Tabs for Images)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.axes = {}
        self.canvases = {}
        self.figures = {}

        tab_names = [
            "Hi-Res Original", "Hi-Res Standard", "Hi-Res SI",
            "Lo-Res Original", "Lo-Res Standard", "Lo-Res SI"
        ]

        for name in tab_names:
            frame = tk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            
            fig = Figure(figsize=(6, 5), dpi=100)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            self.figures[name] = fig
            self.axes[name] = ax
            self.canvases[name] = canvas

        # Apply initial theme
        self.apply_theme()

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.btn_theme.config(text="☀️ Toggle Light Mode" if self.is_dark_mode else "🌙 Toggle Dark Mode")
        self.apply_theme()

    def apply_theme(self):
        theme = self.themes["dark"] if self.is_dark_mode else self.themes["light"]
        
        self.config(bg=theme["bg"])
        self.control_frame.config(bg=theme["bg"])
        
        for widget in self.label_widgets:
            widget.config(bg=theme["bg"], fg=theme["fg"])
            
        for widget in self.entry_widgets:
            if isinstance(widget, tk.Entry):
                widget.config(bg=theme["entry_bg"], fg=theme["entry_fg"], insertbackground=theme["fg"])
            elif isinstance(widget, tk.Frame):
                widget.config(bg=theme["bg"])
                
        self.txt_guide.config(bg=theme["entry_bg"], fg=theme["entry_fg"])

        # Update Matplotlib plots
        for name, fig in self.figures.items():
            fig.patch.set_facecolor(theme["plot_bg"])
            self.axes[name].set_facecolor(theme["plot_bg"])
            self.canvases[name].draw()

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")])
        if path:
            self.img_bgr = cv2.imread(path)
            if self.img_bgr is not None:
                self.lbl_file.config(text=os.path.basename(path))
                self.lbl_status.config(text="Status: Image Loaded. Awaiting Run.")
            else:
                messagebox.showerror("Error", "Could not read the image.")

    def run_detection(self):
        if self.img_bgr is None:
            messagebox.showwarning("Notice", "Please upload an image first.")
            return
        
        try:
            # Pull inputs from text boxes
            harris_k     = self.params["harris_k"].get()
            hi_thresh    = self.params["hi_thresh"].get()
            lo_thresh    = self.params["lo_thresh"].get()
            si_hi_thresh = self.params["si_hi_thresh"].get()
            si_lo_thresh = self.params["si_lo_thresh"].get()
            dog_thresh   = self.params["dog_thresh"].get()
            scale_factor = self.params["scale_factor"].get()
            nms_radius   = self.params["nms_radius"].get()
            num_scales   = self.params["num_scales"].get()
        except tk.TclError:
            messagebox.showerror("Input Error", "Please ensure all parameters are valid numbers.")
            return

        self.lbl_status.config(text="Status: Processing...")
        self.update_idletasks() # Refresh UI

        # ── Pipeline Execution ──────────────────────────────────────────
        hi_bgr  = self.img_bgr.copy()
        hi_gray = cv2.cvtColor(hi_bgr, cv2.COLOR_BGR2GRAY)
        hi_f32  = preprocess(hi_gray, is_low_res=False)

        lo_bgr  = simulate_low_res(hi_bgr, scale_factor)
        lo_gray = cv2.cvtColor(lo_bgr, cv2.COLOR_BGR2GRAY)
        lo_f32  = preprocess(lo_gray, is_low_res=True)

        hi_std = standard_harris(hi_f32, HARRIS_BLOCK_SIZE, HARRIS_KSIZE, harris_k, hi_thresh, nms_radius)
        hi_si  = scale_invariant_harris_improved(hi_f32, num_scales, SCALE_SIGMA_BASE, HARRIS_BLOCK_SIZE, harris_k, si_hi_thresh, nms_radius, dog_thresh=dog_thresh)
        lo_std = standard_harris(lo_f32, HARRIS_BLOCK_SIZE, HARRIS_KSIZE, harris_k, lo_thresh, nms_radius)
        lo_si  = scale_invariant_harris_improved(lo_f32, num_scales, SCALE_SIGMA_BASE, HARRIS_BLOCK_SIZE, harris_k, si_lo_thresh, nms_radius, dog_thresh=dog_thresh)

        # ── Rendering logic ─────────────────────────────────────────────
        def to_rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res_hi_std = draw_corners(hi_bgr, hi_std, (0, 255, 80))
        res_hi_si  = draw_corners(hi_bgr, hi_si, (0, 180, 255))
        res_lo_std = draw_corners(lo_bgr, lo_std, (0, 255, 80))
        res_lo_si  = draw_corners(lo_bgr, lo_si, (0, 180, 255))

        images_dict = {
            "Hi-Res Original": to_rgb(hi_bgr),
            "Hi-Res Standard": to_rgb(res_hi_std),
            "Hi-Res SI":       to_rgb(res_hi_si),
            "Lo-Res Original": to_rgb(lo_bgr),
            "Lo-Res Standard": to_rgb(res_lo_std),
            "Lo-Res SI":       to_rgb(res_lo_si)
        }

        # Update tabs natively
        for name, img_data in images_dict.items():
            ax = self.axes[name]
            ax.clear()
            ax.imshow(img_data)
            ax.axis('off')
            self.canvases[name].draw()

        self.lbl_status.config(text="Status: Done! Check Tabs.")

if __name__ == "__main__":
    app = HarrisApp()
    app.mainloop()