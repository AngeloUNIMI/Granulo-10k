from __future__ import annotations

import argparse
import ctypes
import random
import re
import time
import webbrowser
from ctypes import wintypes
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk, ImageFilter

from inference import StrandPredictor, resolve_view_pair
from overlay import annotate_measurements


APP_TITLE = "Granulo-10k - Multimodal Strand Measurement Demo"
DATASET_URL = "https://github.com/AngeloUNIMI/Granulo-10k"
HEADER_BG = "#f5efe3"
HEADER_FG = "#111827"
HEADER_BLUE = "#0b4f7a"

# Horizontal conference-header margin, expressed as a percentage of the
# available header width. The same value is used on both sides:
#   - UNIMI logo: left margin
#   - QR code: right margin
#
# Example:
#   5.0  -> elements closer to the window edges
#   8.0  -> moderate inset
#   12.0 -> elements further toward the center
HEADER_SIDE_MARGIN_PERCENT = 3.0

# Embedded point-cloud automatic rotation.
# With 1 degree every 100 ms, one full revolution takes about 36 seconds.
PC_ROTATION_ENABLED = True
PC_ROTATION_INTERVAL_MS = 100
PC_ROTATION_DEGREES_PER_STEP = 1.0
PC_ROTATION_ELEVATION = 20.0
PC_ROTATION_START_AZIMUTH = 30.0


def enable_high_dpi_awareness():
    """
    Make Tkinter render natively on high-DPI Windows displays instead of
    letting Windows bitmap-scale the application.
    """
    try:
        # Windows 10/11: Per-Monitor DPI Aware V2.
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4)
        )
        return
    except Exception:
        pass

    try:
        # Windows 8.1+ fallback: PROCESS_PER_MONITOR_DPI_AWARE == 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass

    try:
        # Older Windows fallback.
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# IMPORTANT: this must run before the first tk.Tk() window is created.
enable_high_dpi_awareness()


try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "unimi.granulo10k.demo"
    )
except Exception:
    pass


def load_measurements(
    measurements_file: Path,
):
    measurements = {}

    with measurements_file.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        for line in f:
            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                strand_id = int(parts[0])
                height = float(parts[1])
                width = float(parts[2])
                thickness = float(parts[3])
            except ValueError:
                continue

            measurements[strand_id] = {
                "height_mm": height,
                "width_mm": width,
                "thickness_mm": thickness,
            }

    return measurements


class StrandDemoGUI(tk.Tk):
    def __init__(
        self,
        checkpoint=None,
        image=None,
    ):
        super().__init__()

        # Synchronize Tk font/widget scaling with the monitor DPI reported
        # after enabling per-monitor DPI awareness.
        try:
            dpi = float(self.winfo_fpixels("1i"))
            self.tk.call("tk", "scaling", dpi / 72.0)
        except Exception:
            pass

        self.title(APP_TITLE)
        self._configure_icon()
        self._configure_window_geometry()

        self.predictor = None
        self.original_image = None
        self.view_a_image = None
        self.view_b_image = None
        self.view_a_display = None
        self.view_b_display = None
        self.display_image = None

        self.tk_image_a = None
        self.tk_image_b = None

        self.mask_a_image = None
        self.mask_b_image = None

        self.image_path = None
        self.mask_path = None
        self.mask_image = None

        self.point_cloud_path = None
        self.point_cloud_candidates = []

        # Embedded 3D point-cloud rotation state.
        self.pc_ax = None
        self.pc_rotation_after_id = None
        self.pc_rotation_angle = PC_ROTATION_START_AZIMUTH

        self.ground_truth = None
        self.last_prediction = None
        self._measurements_cache = {}

        # Prediction history for the current GUI session.
        # One entry is appended after every successful prediction.
        self.prediction_history = []

        self.history_window = None
        self.history_figure = None
        self.history_canvas = None
        self.history_axes = None
        self.history_count_text = tk.StringVar(
            value="Predictions in this session: 0"
        )

        # Optional user-defined reference values, in millimetres.
        # Empty string means "no reference line".
        # Default reference granulometry values [mm].
        self.history_ref_height = tk.StringVar(value="115")
        self.history_ref_width = tk.StringVar(value="20")
        self.history_ref_thickness = tk.StringVar(value="0.7")

        # Optional exponential moving-average (EMA) smoothing for the
        # history plot. The user specifies a span; internally:
        #
        #     alpha = 2 / (span + 1)
        #
        # A larger span produces stronger/smoother filtering.
        self.history_smoothing_enabled = tk.BooleanVar(
            value=False
        )
        self.history_ema_span = tk.IntVar(
            value=10
        )

        self.acquisition_paths = []
        self.acquisition_index = None
        self.all_acquisition_paths_cache = None

        self._thickness_sets = {}

        self.autoplay_active = False
        self.autoplay_after_id = None
        self.autoplay_interval_ms = 2000
        self.autoplay_pool = []
        self.autoplay_index = 0

        self.show_mask_outline = tk.BooleanVar(
            value=False
        )

        self.model_info_expanded = False

        self.model_name = tk.StringVar(
            value="Model: not loaded"
        )

        self.input_pair_name = tk.StringVar(
            value="Paired views: not available"
        )

        self.view_a_title = tk.StringVar(
            value="View A"
        )

        self.view_b_title = tk.StringVar(
            value="View B"
        )

        self.image_caption_a = tk.StringVar(
            value="Image encoder input + measurement overlay"
        )

        self.image_caption_b = tk.StringVar(
            value="Image encoder input"
        )

        self.point_cloud_name = tk.StringVar(
            value="Point cloud — not available"
        )

        self.input_mode_text = tk.StringVar(
            value="NO INPUT"
        )

        self.inference_time_text = tk.StringVar(
            value="Inference: —"
        )

        self.model_info_text = tk.StringVar(
            value="Model information unavailable"
        )

        self.orientation = tk.StringVar(
            value="frontal"
        )

        self.status = tk.StringVar(
            value=(
                "Load the multimodal model and select "
                "a strand image."
            )
        )

        # Predictions
        self.height_value = tk.StringVar(value="—")
        self.width_value = tk.StringVar(value="—")
        self.thickness_value = tk.StringVar(value="—")

        # Ground truth
        self.gt_height_value = tk.StringVar(value="—")
        self.gt_width_value = tk.StringVar(value="—")
        self.gt_thickness_value = tk.StringVar(value="—")

        # Errors
        self.error_height_value = tk.StringVar(value="—")
        self.error_width_value = tk.StringVar(value="—")
        self.error_thickness_value = tk.StringVar(value="—")

        self.header_expanded = True
        self.header_frame = None
        self.collapsed_header_frame = None
        self.header_toggle_button = None
        self.header_unimi_photo = None
        self.header_icip_photo = None
        self.header_qr_photo = None

        self._build_ui()

        self.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

        if checkpoint:
            self.load_model(
                Path(checkpoint)
            )

        if image:
            startup_image = Path(image)

            if startup_image.exists():
                # Defer image loading until Tk has completed the initial
                # geometry pass, so canvases and point-cloud preview have
                # valid dimensions.
                self.after(
                    0,
                    lambda p=startup_image: self._load_image_path(
                        p,
                        source="Startup image",
                    ),
                )
            else:
                self.after(
                    0,
                    lambda p=startup_image: messagebox.showerror(
                        "Image error",
                        f"Startup image not found:\n{p}",
                    ),
                )

    def _on_close(self):
        self._stop_autoplay()
        self._stop_point_cloud_rotation()
        self.destroy()

    def _configure_icon(self):
        icon_ico = Path(__file__).with_name(
            "granulo_demo_icon.ico"
        )
        icon_png = Path(__file__).with_name(
            "granulo_demo_icon.png"
        )

        if icon_ico.exists():
            try:
                self.iconbitmap(
                    default=str(icon_ico)
                )
            except Exception:
                pass

        if icon_png.exists():
            try:
                self._taskbar_icon = (
                    ImageTk.PhotoImage(
                        Image.open(icon_png).convert(
                            "RGBA"
                        )
                    )
                )

                self.iconphoto(
                    True,
                    self._taskbar_icon,
                )
            except Exception:
                pass

    def _configure_window_geometry(self):
        """Open at 5/6 of the usable screen and center it."""
        self.update_idletasks()

        try:
            rect = wintypes.RECT()

            SPI_GETWORKAREA = 0x0030

            ok = ctypes.windll.user32.SystemParametersInfoW(
                SPI_GETWORKAREA,
                0,
                ctypes.byref(rect),
                0,
            )

            if not ok:
                raise OSError(
                    "Could not query Windows work area."
                )

            left = rect.left
            top = rect.top
            work_width = rect.right - rect.left
            work_height = rect.bottom - rect.top

        except Exception:
            left = 0
            top = 0
            work_width = self.winfo_screenwidth()
            work_height = self.winfo_screenheight()

        window_width = int(
            work_width * 5 / 6
        )
        window_height = int(
            work_height * 5 / 6
        )

        x = (
            left
            + (work_width - window_width) // 2
        )
        y = (
            top
            + (work_height - window_height) // 2
        )

        self.geometry(
            f"{window_width}x{window_height}+{x}+{y}"
        )

        self.minsize(
            1000,
            700,
        )

    @staticmethod
    def _asset_path(filename: str) -> Path:
        """
        Resolve conference assets using both the packaged names and the
        original filenames used by the poster/demo assets.
        """
        base = Path(__file__).resolve().parent

        aliases = {
            "unimi_logo.png": (
                "unimi_logo.png",
                "minerva2011.png",
            ),
            "icip2026_logo.png": (
                "icip2026_logo.png",
                "icip_logo.png",
            ),
            "qrcode.png": (
                "qrcode.png",
                "qrcode(1).png",
            ),
        }

        names = aliases.get(
            filename,
            (filename,),
        )

        for folder in (
            base / "assets",
            base,
        ):
            for name in names:
                candidate = folder / name

                if candidate.exists():
                    return candidate

        # Return the preferred path so the caller still gets a useful
        # FileNotFoundError if no asset exists.
        return base / "assets" / filename

    @staticmethod
    def _photo_from_asset(
        path: Path,
        max_size: tuple[int, int],
        crop_white_margin: bool = False,
    ):
        image = Image.open(path).convert("RGBA")

        if crop_white_margin:
            rgb = image.convert("RGB")
            array = np.asarray(rgb)

            # Locate everything that is not essentially white. This is useful
            # for the QR image, which contains a generous white border.
            nonwhite = np.any(
                array < 248,
                axis=2,
            )

            ys, xs = np.where(nonwhite)

            if len(xs) and len(ys):
                left = max(int(xs.min()) - 8, 0)
                top = max(int(ys.min()) - 8, 0)
                right = min(int(xs.max()) + 9, image.width)
                bottom = min(int(ys.max()) + 9, image.height)

                image = image.crop(
                    (left, top, right, bottom)
                )

        image.thumbnail(
            max_size,
            Image.Resampling.LANCZOS,
        )

        return ImageTk.PhotoImage(image)

    def _build_conference_header(self):
        header = tk.Frame(self, bg=HEADER_BG, bd=0, highlightthickness=0)
        self.header_frame = header
        header.grid(row=0, column=0, columnspan=2, sticky="ew")

        # Responsive symmetric layout:
        # left-margin | UNIMI | center information | QR | right-margin
        #
        # Columns 0 and 4 are resized to HEADER_SIDE_MARGIN_PERCENT of the
        # actual header width. This gives the logo and QR equal outer margins.
        header.columnconfigure(0, weight=0, minsize=0)
        header.columnconfigure(1, weight=0)
        header.columnconfigure(2, weight=1)
        header.columnconfigure(3, weight=0)
        header.columnconfigure(4, weight=0, minsize=0)

        header.bind(
            "<Configure>",
            self._update_header_side_margins,
            add="+",
        )

        try:
            self.header_unimi_photo = self._photo_from_asset(
                self._asset_path("unimi_logo.png"), (256, 256)
            )
            self.header_unimi_label = tk.Label(
                header, image=self.header_unimi_photo, bg=HEADER_BG, bd=0
            )
        except Exception:
            self.header_unimi_label = tk.Label(
                header,
                text="Università degli Studi di Milano",
                bg=HEADER_BG,
                fg=HEADER_BLUE,
                font=("Segoe UI", 10, "bold"),
            )

        self.header_unimi_label.grid(
            row=0,
            column=1,
            rowspan=2,
            padx=(0, 22),
            pady=8,
            sticky="w",
        )

        self.header_center = tk.Frame(header, bg=HEADER_BG, bd=0)
        self.header_center.grid(
            row=0,
            column=2,
            rowspan=2,
            sticky="nsew",
            pady=(6, 4),
        )

        tk.Label(
            self.header_center,
            text="GRANULO-10K",
            bg=HEADER_BG,
            fg=HEADER_FG,
            font=("Segoe UI", 23, "bold"),
        ).pack()

        tk.Label(
            self.header_center,
            text=(
                "A LARGE-SCALE BENCHMARK DATASET FOR MULTIPLE-VIEW "
                "INDUSTRIAL GRANULOMETRY"
            ),
            bg=HEADER_BG,
            fg=HEADER_FG,
            font=("Segoe UI", 14, "bold"),
            justify="center",
        ).pack(pady=(0, 2))

        tk.Label(
            self.header_center,
            text="Pasquale Coscia, Angelo Genovese, Vincenzo Piuri, Fabio Scotti",
            bg=HEADER_BG,
            fg=HEADER_FG,
            font=("Segoe UI", 11, "italic"),
        ).pack()

        tk.Label(
            self.header_center,
            text=(
                "Department of Computer Science, "
                "Università degli Studi di Milano, Italy"
            ),
            bg=HEADER_BG,
            fg=HEADER_FG,
            font=("Segoe UI", 10, "italic"),
        ).pack(pady=(0, 1))

        try:
            self.header_icip_photo = self._photo_from_asset(
                self._asset_path("icip2026_logo.png"), (244, 104)
            )
            tk.Label(
                self.header_center,
                image=self.header_icip_photo,
                bg=HEADER_BG,
                bd=0,
            ).pack(pady=(1, 0))
        except Exception:
            tk.Label(
                self.header_center,
                text="ICIP 2026 · Tampere",
                bg=HEADER_BG,
                fg=HEADER_BLUE,
                font=("Segoe UI", 9, "bold"),
            ).pack()

        self.header_qr_holder = tk.Frame(header, bg=HEADER_BG, bd=0)
        self.header_qr_holder.grid(
            row=0,
            column=3,
            rowspan=2,
            padx=(22, 0),
            pady=(6, 5),
            sticky="e",
        )

        try:
            self.header_qr_photo = self._photo_from_asset(
                self._asset_path("qrcode.png"),
                (200, 200),
                crop_white_margin=True,
            )
            qr_label = tk.Label(
                self.header_qr_holder,
                image=self.header_qr_photo,
                bg=HEADER_BG,
                cursor="hand2",
                bd=0,
            )
            qr_label.pack()
            qr_label.bind(
                "<Button-1>", lambda _event: webbrowser.open(DATASET_URL)
            )
            scan = tk.Label(
                self.header_qr_holder,
                text="Scan or click",
                bg=HEADER_BG,
                fg=HEADER_BLUE,
                cursor="hand2",
                font=("Segoe UI", 8, "bold"),
            )
            scan.pack(pady=(1, 0))
            scan.bind("<Button-1>", lambda _event: webbrowser.open(DATASET_URL))
        except Exception:
            tk.Button(
                self.header_qr_holder,
                text="Dataset & Code",
                command=lambda: webbrowser.open(DATASET_URL),
            ).pack()

        self.header_toggle_button = tk.Button(
            header,
            text="▲",
            command=self.toggle_conference_header,
            bg=HEADER_BG,
            fg="#444444",
            activebackground=HEADER_BG,
            relief="flat",
            bd=0,
            padx=8,
            pady=2,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        # Overlay the collapse control at the top-right so it does not
        # participate in the grid and therefore does not distort the QR margin.
        self.header_toggle_button.place(
            relx=1.0,
            x=-8,
            y=6,
            anchor="ne",
        )

        self.header_separator = tk.Frame(header, bg="#d7d0c4", height=1)
        self.header_separator.grid(
            row=2,
            column=0,
            columnspan=5,
            sticky="ew",
        )

        # Compact replacement shown when the full conference banner is hidden.
        # It is a sibling of header_frame, so the large header frame can be
        # removed from the root grid completely and therefore occupies no space.
        self.collapsed_header_frame = tk.Frame(
            self,
            bg=HEADER_BG,
            bd=0,
            highlightthickness=0,
        )

        collapsed_button = tk.Button(
            self.collapsed_header_frame,
            text="▼  GRANULO-10K · ICIP 2026",
            command=self.toggle_conference_header,
            bg=HEADER_BG,
            fg="#444444",
            activebackground=HEADER_BG,
            relief="flat",
            bd=0,
            padx=8,
            pady=2,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        collapsed_button.pack(
            side="right",
            padx=8,
            pady=3,
        )

        # Hidden initially; the full banner is shown at startup.
        self.collapsed_header_frame.grid_remove()

    def _update_header_side_margins(self, _event=None):
        """
        Keep the UNIMI left margin and QR right margin equal.

        HEADER_SIDE_MARGIN_PERCENT is interpreted as a percentage of the
        current conference-header width, so the layout remains proportional
        when the window is resized.
        """
        if self.header_frame is None:
            return

        width = self.header_frame.winfo_width()

        # Ignore Tk's transient 1-pixel geometry during initial construction.
        if width <= 1:
            return

        percent = max(
            0.0,
            min(
                float(HEADER_SIDE_MARGIN_PERCENT),
                30.0,
            ),
        )

        margin_px = int(
            round(
                width * percent / 100.0
            )
        )

        self.header_frame.columnconfigure(
            0,
            minsize=margin_px,
        )
        self.header_frame.columnconfigure(
            4,
            minsize=margin_px,
        )

    def toggle_conference_header(self):
        if self.header_frame is None:
            return

        self.header_expanded = not self.header_expanded

        if self.header_expanded:
            # Remove the compact bar and restore the full conference header.
            if self.collapsed_header_frame is not None:
                self.collapsed_header_frame.grid_remove()

            self.header_frame.grid(
                row=0,
                column=0,
                columnspan=2,
                sticky="ew",
            )

            self.header_toggle_button.configure(
                text="▲"
            )

            self._update_header_side_margins()

        else:
            # Remove the ENTIRE expanded header from the root grid.
            # Hiding only its children leaves the header frame's old requested
            # height behind, which is what caused the large blank strip.
            self.header_frame.grid_remove()

            if self.collapsed_header_frame is not None:
                self.collapsed_header_frame.grid(
                    row=0,
                    column=0,
                    columnspan=2,
                    sticky="ew",
                )

        self.update_idletasks()

        if hasattr(self, "canvas_a"):
            self.refresh_view_canvases()

        if hasattr(self, "pc_preview_frame"):
            self._refresh_point_cloud_geometry()

    def _build_ui(self):
        self.columnconfigure(
            1,
            weight=1,
        )

        self.rowconfigure(
            1,
            weight=1,
        )

        self._build_conference_header()

        controls = ttk.Frame(
            self,
            padding=18,
        )

        controls.grid(
            row=1,
            column=0,
            sticky="ns",
        )

        viewer = ttk.Frame(
            self,
            padding=(0, 18, 18, 18),
        )

        self.viewer = viewer

        viewer.grid(
            row=1,
            column=1,
            sticky="nsew",
        )

        viewer.rowconfigure(
            1,
            weight=0,
            minsize=480,
        )

        # The point-cloud preview expands into all remaining vertical space.
        viewer.rowconfigure(
            4,
            weight=1,
            minsize=160,
        )

        viewer.columnconfigure(
            0,
            weight=1,
        )

        viewer.columnconfigure(
            1,
            weight=1,
        )

        ttk.Label(
            controls,
            text="Multimodal measurement controls",
            font=("Segoe UI", 10, "bold"),
            foreground="#444444",
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        ttk.Button(
            controls,
            text="Load model...",
            command=self.choose_model,
        ).pack(
            fill="x",
            pady=4,
        )

        ttk.Label(
            controls,
            textvariable=self.model_name,
            font=("Segoe UI", 9),
            foreground="#555555",
        ).pack(
            anchor="w",
            pady=(4, 4),
        )

        self.model_info_button = ttk.Button(
            controls,
            text="Model information ▸",
            command=self.toggle_model_information,
        )

        self.model_info_button.pack(
            fill="x",
            pady=(0, 2),
        )

        self.model_info_frame = ttk.Frame(
            controls,
        )
        # Do not pack the frame initially. It is inserted only when the
        # Model information section is expanded, so no empty vertical space
        # remains when the section is collapsed.

        self.model_info_label = ttk.Label(
            self.model_info_frame,
            textvariable=self.model_info_text,
            font=("Segoe UI", 8),
            foreground="#555555",
            justify="left",
            wraplength=285,
        )
        self.model_info_label.pack(
            fill="x",
            pady=(2, 6),
        )

        self.select_image_button = ttk.Button(
            controls,
            text="Select strand image...",
            command=self.choose_image,
        )
        self.select_image_button.pack(
            fill="x",
            pady=4,
        )

        nav_frame = ttk.Frame(
            controls,
        )
        nav_frame.pack(
            fill="x",
            pady=(2, 4),
        )

        self.prev_button = ttk.Button(
            nav_frame,
            text="◀ Previous",
            command=self.previous_acquisition,
            state="disabled",
        )
        self.prev_button.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 2),
        )

        self.next_button = ttk.Button(
            nav_frame,
            text="Next ▶",
            command=self.next_acquisition,
            state="disabled",
        )
        self.next_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(2, 0),
        )

        nav_frame.columnconfigure(
            0,
            weight=1,
        )
        nav_frame.columnconfigure(
            1,
            weight=1,
        )

        demo_frame = ttk.Frame(
            controls,
        )
        demo_frame.pack(
            fill="x",
            pady=(0, 4),
        )

        ttk.Button(
            demo_frame,
            text="Random acquisition",
            command=self.random_acquisition,
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 2),
        )

        self.autoplay_button = ttk.Button(
            demo_frame,
            text="Start demo",
            command=self.toggle_autoplay,
        )
        self.autoplay_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(2, 0),
        )

        demo_frame.columnconfigure(
            0,
            weight=1,
        )
        demo_frame.columnconfigure(
            1,
            weight=1,
        )

        ttk.Label(
            controls,
            textvariable=self.input_pair_name,
            font=("Segoe UI", 9),
            foreground="#555555",
        ).pack(
            anchor="w",
            pady=(4, 2),
        )

        self.point_cloud_button = ttk.Button(
            controls,
            text="View point cloud...",
            command=self.view_point_cloud,
            state="disabled",
        )

        self.point_cloud_button.pack(
            fill="x",
            pady=4,
        )

        ttk.Label(
            controls,
            textvariable=self.point_cloud_name,
            font=("Segoe UI", 9),
            foreground="#555555",
        ).pack(
            anchor="w",
            pady=(4, 4),
        )

        badge_frame = ttk.Frame(
            controls,
        )
        badge_frame.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            badge_frame,
            text="Input mode:",
        ).pack(
            side="left",
        )

        self.input_mode_badge = tk.Label(
            badge_frame,
            textvariable=self.input_mode_text,
            padx=8,
            pady=2,
            fg="white",
            bg="#666666",
            font=("Segoe UI", 8, "bold"),
        )
        self.input_mode_badge.pack(
            side="right",
        )

        ttk.Separator(
            controls
        ).pack(
            fill="x",
            pady=14,
        )

        ttk.Label(
            controls,
            text="Acquisition view",
            font=("Segoe UI", 11, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 6),
        )

        ttk.Radiobutton(
            controls,
            text="Frontal",
            variable=self.orientation,
            value="frontal",
            command=self._orientation_changed,
        ).pack(
            anchor="w"
        )

        ttk.Radiobutton(
            controls,
            text="Sideways",
            variable=self.orientation,
            value="sideways",
            command=self._orientation_changed,
        ).pack(
            anchor="w"
        )

        ttk.Checkbutton(
            controls,
            text="Show segmentation outline",
            variable=self.show_mask_outline,
            command=self._mask_outline_changed,
        ).pack(
            anchor="w",
            pady=(6, 0),
        )

        self.run_button = ttk.Button(
            controls,
            text="Run multimodal prediction",
            command=self.run_prediction,
        )

        self.run_button.pack(
            fill="x",
            pady=(14, 5),
        )

        ttk.Button(
            controls,
            text="Save annotated image...",
            command=self.save_image,
        ).pack(
            fill="x",
            pady=5,
        )

        self.prediction_rows = self._add_measurement_section(
            controls,
            "Predicted measurements",
            [
                ("height", "Height", self.height_value),
                ("width", "Width", self.width_value),
                ("thickness", "Thickness", self.thickness_value),
            ],
        )

        self.ground_truth_rows = self._add_measurement_section(
            controls,
            "Ground truth measurements",
            [
                ("height", "Height", self.gt_height_value),
                ("width", "Width", self.gt_width_value),
                ("thickness", "Thickness", self.gt_thickness_value),
            ],
        )

        self.error_rows = self._add_measurement_section(
            controls,
            "Prediction error",
            [
                ("height", "Δ Height", self.error_height_value),
                ("width", "Δ Width", self.error_width_value),
                ("thickness", "Δ Thickness", self.error_thickness_value),
            ],
        )

        self._update_visible_measurement_rows()

        ttk.Separator(
            controls
        ).pack(
            fill="x",
            pady=14,
        )

        self.device_label = ttk.Label(
            controls,
            text=(
                f"CUDA available: "
                f"{torch.cuda.is_available()}"
            ),
        )

        self.device_label.pack(
            anchor="w",
        )

        ttk.Label(
            controls,
            textvariable=self.inference_time_text,
            font=("Segoe UI", 9),
        ).pack(
            anchor="w",
            pady=(4, 4),
        )

        self.gates_button = ttk.Button(
            controls,
            text="MMoE gate activity...",
            command=self.show_gate_visualization,
            state="disabled",
        )
        self.gates_button.pack(
            fill="x",
            pady=(2, 0),
        )

        self.history_button = ttk.Button(
            controls,
            text="Prediction history...",
            command=self.show_prediction_history,
            state="disabled",
        )
        self.history_button.pack(
            fill="x",
            pady=(5, 0),
        )

        ttk.Label(
            viewer,
            textvariable=self.view_a_title,
            font=("Segoe UI", 11, "bold"),
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=(0, 6),
        )

        ttk.Label(
            viewer,
            textvariable=self.view_b_title,
            font=("Segoe UI", 11, "bold"),
        ).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 0),
            pady=(0, 6),
        )

        self.canvas_a = tk.Canvas(
            viewer,
            background="#202124",
            highlightthickness=1,
            highlightbackground="#5f6368",
            height=480,
        )

        self.canvas_a.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 8),
        )

        self.canvas_a.bind(
            "<Configure>",
            lambda _event: self.refresh_view_canvases(),
        )

        self.canvas_b = tk.Canvas(
            viewer,
            background="#202124",
            highlightthickness=1,
            highlightbackground="#5f6368",
            height=480,
        )

        self.canvas_b.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(8, 0),
        )

        self.canvas_b.bind(
            "<Configure>",
            lambda _event: self.refresh_view_canvases(),
        )

        ttk.Label(
            viewer,
            textvariable=self.image_caption_a,
            anchor="center",
            foreground="#555555",
        ).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=(0, 8),
            pady=(5, 10),
        )

        ttk.Label(
            viewer,
            textvariable=self.image_caption_b,
            anchor="center",
            foreground="#555555",
        ).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(5, 10),
        )

        pc_header = ttk.Frame(
            viewer,
        )

        pc_header.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 6),
        )

        pc_header.columnconfigure(
            0,
            weight=1,
        )

        self.pc_preview_label = ttk.Label(
            pc_header,
            textvariable=self.point_cloud_name,
            font=("Segoe UI", 10, "bold"),
        )

        self.pc_preview_label.grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Button(
            pc_header,
            text="Open 3D",
            command=self.view_point_cloud,
        ).grid(
            row=0,
            column=1,
            padx=(8, 4),
        )

        self.mask_button = ttk.Button(
            pc_header,
            text="Open segmentation mask",
            command=self.open_segmentation_mask,
            state="disabled",
        )

        self.mask_button.grid(
            row=0,
            column=2,
            padx=(4, 0),
        )

        self.pc_preview_frame = ttk.Frame(
            viewer,
            height=160,
        )

        self.pc_preview_frame.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="nsew",
        )

        self.pc_preview_frame.bind(
            "<Configure>",
            self._on_point_cloud_frame_resize,
        )

        self.pc_preview_placeholder = ttk.Label(
            self.pc_preview_frame,
            text="Select a strand image to preview the point cloud",
            anchor="center",
        )

        self.pc_preview_placeholder.pack(
            fill="both",
            expand=True,
        )

        self.pc_figure = None
        self.pc_figure_canvas = None
        self.pc_ax = None

        ttk.Label(
            viewer,
            textvariable=self.status,
            anchor="w",
        ).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )

    @staticmethod
    def _add_measurement_section(
        parent,
        title,
        rows,
    ):
        ttk.Separator(
            parent
        ).pack(
            fill="x",
            pady=14,
        )

        ttk.Label(
            parent,
            text=title,
            font=("Segoe UI", 11, "bold"),
        ).pack(
            anchor="w",
        )

        frame = ttk.Frame(
            parent,
            padding=(0, 8),
        )

        frame.pack(
            fill="x",
        )

        row_widgets = {}

        for row_index, (
            key,
            label,
            variable,
        ) in enumerate(rows):
            name_label = ttk.Label(
                frame,
                text=label,
            )

            value_label = ttk.Label(
                frame,
                textvariable=variable,
                font=("Segoe UI", 11, "bold"),
            )

            name_label.grid(
                row=row_index,
                column=0,
                sticky="w",
                pady=3,
            )

            value_label.grid(
                row=row_index,
                column=1,
                sticky="e",
                padx=(20, 0),
            )

            row_widgets[key] = (
                name_label,
                value_label,
            )

        frame.columnconfigure(
            1,
            weight=1,
        )

        return row_widgets

    def _update_visible_measurement_rows(self):
        """
        The multimodal model predicts H, W and T internally for every sample,
        but the GUI displays only dimensions that are physically observable
        from the selected acquisition view.
        """
        visible = {
            "height",
            "width" if self.orientation.get() == "frontal" else "thickness",
        }

        for row_group in (
            self.prediction_rows,
            self.ground_truth_rows,
            self.error_rows,
        ):
            for key, widgets in row_group.items():
                for widget in widgets:
                    if key in visible:
                        widget.grid()
                    else:
                        widget.grid_remove()

    def choose_model(self):
        filename = filedialog.askopenfilename(
            title="Select multimodal checkpoint",
            filetypes=[
                (
                    "PyTorch checkpoint",
                    "*.pt *.pth",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if filename:
            self.load_model(
                Path(filename)
            )

    def load_model(
        self,
        path: Path,
    ):
        try:
            self.status.set(
                f"Loading multimodal model: {path.name}"
            )

            self.update_idletasks()

            self.predictor = StrandPredictor(
                path
            )

            self.model_name.set(
                f"Model: {path.name}"
            )

            backbone = self.predictor.backbone_label

            self.image_caption_a.set(
                f"{backbone} input + measurement overlay"
            )
            self.image_caption_b.set(
                f"{backbone} input"
            )

            epoch_text = (
                f", best epoch {self.predictor.best_epoch}"
                if self.predictor.best_epoch is not None
                else ""
            )

            self.status.set(
                "Multimodal model loaded "
                f"({self.predictor.device}{epoch_text})."
            )

            self.device_label.configure(
                text=(
                    f"Model device: "
                    f"{self.predictor.device}"
                )
            )

            self._update_model_information()

        except Exception as exc:
            self.predictor = None

            self.model_name.set(
                "Model: not loaded"
            )

            messagebox.showerror(
                "Model error",
                str(exc),
            )

            self.status.set(
                "Could not load multimodal model."
            )

    def choose_image(self):
        self._stop_autoplay()

        filename = filedialog.askopenfilename(
            title="Select strand image",
            filetypes=[
                (
                    "Images",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not filename:
            return

        self._load_image_path(
            Path(filename),
            source="Selected",
        )

    def _load_image_path(
        self,
        path: Path,
        source: str = "Selected",
    ):
        try:
            self.image_path = Path(path)

            image_a_path, image_b_path = resolve_view_pair(
                self.image_path
            )

            # Use View A as the canonical acquisition path for navigation.
            self.image_path = image_a_path

            with Image.open(
                image_a_path
            ) as im:
                self.view_a_image = (
                    im.convert("RGB").copy()
                )

            with Image.open(
                image_b_path
            ) as im:
                self.view_b_image = (
                    im.convert("RGB").copy()
                )

            self.original_image = self.view_a_image
            self.view_a_display = self.view_a_image.copy()
            self.view_b_display = self.view_b_image.copy()
            self.display_image = self.view_a_display

            self.view_a_title.set(
                f"View A — {image_a_path.name}"
            )

            self.view_b_title.set(
                f"View B — {image_b_path.name}"
            )

            self._resize_rgb_row_for_images()

            self._load_masks_for_views(
                image_a_path,
                image_b_path,
            )

            self.mask_image = self.mask_a_image

            # Detect frontal/sideways automatically before choosing a PC.
            self._auto_detect_orientation(
                self.image_path
            )

            self._update_paired_views_label()
            self._find_point_cloud()
            self._load_ground_truth()
            self._build_strand_acquisition_list()
            self._reset_prediction_values()
            self._update_mask_button()
            self._update_input_mode_badge()
            self._rebuild_view_a_display()

            loaded = [
                "paired A/B views",
            ]

            if self.mask_image is not None:
                loaded.append("mask")

            if self.point_cloud_path is not None:
                loaded.append("point cloud")
            else:
                loaded.append("image-only fallback")

            self.status.set(
                f"{source}: "
                + ", ".join(loaded)
                + "."
            )

            self.update_idletasks()
            self.refresh_view_canvases()
            self._refresh_point_cloud_geometry()

            self.after_idle(
                self.refresh_view_canvases
            )

        except Exception as exc:
            messagebox.showerror(
                "Image error",
                str(exc),
            )


    def toggle_model_information(self):
        self.model_info_expanded = (
            not self.model_info_expanded
        )

        if self.model_info_expanded:
            # Insert the whole frame immediately before the image selector.
            self.model_info_frame.pack(
                fill="x",
                before=self.select_image_button,
            )
            self.model_info_button.configure(
                text="Model information ▾"
            )
        else:
            # Remove the whole frame so it occupies exactly zero space.
            self.model_info_frame.pack_forget()
            self.model_info_button.configure(
                text="Model information ▸"
            )

        # Let Tk recompute the sidebar geometry immediately.
        self.update_idletasks()

    def _update_model_information(self):
        if self.predictor is None:
            self.model_info_text.set(
                "Model information unavailable"
            )
            return

        test = (
            self.predictor.test_metrics
            or {}
        )

        h = test.get(
            "mae_height_mm",
            None,
        )
        w = test.get(
            "mae_width_mm",
            None,
        )
        t = test.get(
            "mae_thickness_mm",
            None,
        )

        mae_text = "Test MAE: unavailable"

        if (
            h is not None
            and w is not None
            and t is not None
        ):
            mae_text = (
                f"Test MAE: H {h:.3f} mm | "
                f"W {w:.3f} mm | "
                f"T {t:.4f} mm"
            )

        fold = (
            self.predictor.fold
            if self.predictor.fold is not None
            else "—"
        )

        epoch = (
            self.predictor.best_epoch
            if self.predictor.best_epoch is not None
            else "—"
        )

        self.model_info_text.set(
            f"{self.predictor.backbone_label} + PointNet++ + "
            f"{self.predictor.num_experts}-expert MMoE\n"
            f"Fold: {fold} | Best epoch: {epoch}\n"
            f"{mae_text}"
        )

    def _update_input_mode_badge(
        self,
        actual_mode=None,
    ):
        if self.image_path is None:
            text = "NO INPUT"
            background = "#666666"
        else:
            mode = actual_mode

            if mode is None:
                mode = (
                    "multimodal"
                    if self.point_cloud_path is not None
                    else "image_only"
                )

            if mode == "multimodal":
                text = "FULL MULTIMODAL"
                background = "#2e7d32"
            else:
                text = "IMAGE-ONLY FALLBACK"
                background = "#b26a00"

        self.input_mode_text.set(
            text
        )

        if hasattr(
            self,
            "input_mode_badge",
        ):
            self.input_mode_badge.configure(
                bg=background
            )

    def _find_thickness_file(
        self,
        image_path: Path,
    ) -> Path | None:
        current = image_path.parent

        for _ in range(6):
            candidate = (
                current
                / "strands_ok_for_thickness.txt"
            )

            if candidate.exists():
                return candidate

            current = current.parent

        return None

    def _get_thickness_set(
        self,
        image_path: Path,
    ):
        thickness_file = self._find_thickness_file(
            image_path
        )

        if thickness_file is None:
            return set()

        key = str(
            thickness_file.resolve()
        )

        if key in self._thickness_sets:
            return self._thickness_sets[key]

        acquisitions = set()

        try:
            with thickness_file.open(
                "r",
                encoding="utf-8-sig",
                errors="ignore",
            ) as f:
                for line in f:
                    token = line.strip().split()

                    if not token:
                        continue

                    acquisition = token[0].strip()

                    if re.fullmatch(
                        r"\d{4}_\d{4}",
                        acquisition,
                    ):
                        acquisitions.add(
                            acquisition
                        )
        except OSError:
            acquisitions = set()

        self._thickness_sets[key] = (
            acquisitions
        )

        return acquisitions

    def _orientation_for_path(
        self,
        image_path: Path,
    ) -> str:
        match = re.match(
            r"(\d{4}_\d{4})_[AB]$",
            image_path.stem,
            re.IGNORECASE,
        )

        if not match:
            return "frontal"

        acquisition = match.group(1)

        return (
            "sideways"
            if acquisition
            in self._get_thickness_set(
                image_path
            )
            else "frontal"
        )

    def _auto_detect_orientation(
        self,
        image_path: Path,
    ):
        orientation = self._orientation_for_path(
            image_path
        )

        self.orientation.set(
            orientation
        )

        self._update_visible_measurement_rows()

    def _build_strand_acquisition_list(self):
        self.acquisition_paths = []
        self.acquisition_index = None

        if self.image_path is None:
            self._update_navigation_buttons()
            return

        folder = self.image_path.parent
        valid_suffixes = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }

        acquisitions = []

        for candidate in sorted(
            folder.glob("*_A.*")
        ):
            if (
                not candidate.is_file()
                or candidate.suffix.lower()
                not in valid_suffixes
            ):
                continue

            try:
                image_a, image_b = resolve_view_pair(
                    candidate
                )
            except Exception:
                continue

            if image_a.exists() and image_b.exists():
                acquisitions.append(
                    image_a
                )

        self.acquisition_paths = acquisitions

        current_prefix = (
            self.image_path.stem[:-2]
            if self.image_path.stem.endswith("_A")
            else self.image_path.stem
        )

        for index, candidate in enumerate(
            self.acquisition_paths
        ):
            if candidate.stem[:-2] == current_prefix:
                self.acquisition_index = index
                break

        self._update_navigation_buttons()

    def _update_navigation_buttons(self):
        count = len(
            self.acquisition_paths
        )

        index = self.acquisition_index

        prev_state = (
            "normal"
            if (
                index is not None
                and index > 0
            )
            else "disabled"
        )

        next_state = (
            "normal"
            if (
                index is not None
                and index < count - 1
            )
            else "disabled"
        )

        if hasattr(
            self,
            "prev_button",
        ):
            self.prev_button.configure(
                state=prev_state
            )

        if hasattr(
            self,
            "next_button",
        ):
            self.next_button.configure(
                state=next_state
            )

    def previous_acquisition(self):
        self._stop_autoplay()

        if (
            self.acquisition_index is None
            or self.acquisition_index <= 0
        ):
            return

        self._load_image_path(
            self.acquisition_paths[
                self.acquisition_index - 1
            ],
            source="Previous",
        )

    def next_acquisition(self):
        self._stop_autoplay()

        if (
            self.acquisition_index is None
            or self.acquisition_index
            >= len(self.acquisition_paths) - 1
        ):
            return

        self._load_image_path(
            self.acquisition_paths[
                self.acquisition_index + 1
            ],
            source="Next",
        )

    def _default_images_root(self):
        project_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )

        candidate = (
            project_root
            / "data"
            / "Granulo-10k"
            / "Images"
            / "Strands_compliant"
        )

        if candidate.exists():
            return candidate

        if self.image_path is not None:
            current = self.image_path.parent

            for _ in range(5):
                if (
                    current.name
                    == "Strands_compliant"
                    and current.parent.name
                    == "Images"
                ):
                    return current

                current = current.parent

        return None

    def _get_all_acquisition_paths(self):
        if self.all_acquisition_paths_cache is not None:
            return self.all_acquisition_paths_cache

        root = self._default_images_root()

        if root is None:
            return []

        valid_suffixes = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }

        paths = []

        for candidate in sorted(
            root.glob("*/*_A.*")
        ):
            if (
                candidate.suffix.lower()
                not in valid_suffixes
            ):
                continue

            try:
                image_a, image_b = resolve_view_pair(
                    candidate
                )
            except Exception:
                continue

            if image_a.exists() and image_b.exists():
                paths.append(
                    image_a
                )

        self.all_acquisition_paths_cache = paths
        return paths

    def random_acquisition(self):
        self._stop_autoplay()

        paths = self._get_all_acquisition_paths()

        if not paths:
            messagebox.showwarning(
                "Random acquisition",
                "Could not find the Granulo-10k image directory.",
            )
            return

        self._load_image_path(
            random.choice(paths),
            source="Random",
        )

    def _pc_dir_for_image(
        self,
        image_path: Path,
    ) -> Path | None:
        parts = list(
            image_path.parent.parts
        )

        try:
            images_index = parts.index(
                "Images"
            )
        except ValueError:
            return None

        parts[images_index] = "PCs"

        return Path(
            *parts
        )

    def _path_has_valid_pc(
        self,
        image_path: Path,
    ) -> bool:
        match = re.match(
            r"(\d{4}_\d{4})_[AB]$",
            image_path.stem,
            re.IGNORECASE,
        )

        if not match:
            return False

        pc_dir = self._pc_dir_for_image(
            image_path
        )

        if (
            pc_dir is None
            or not pc_dir.exists()
        ):
            return False

        prefix = match.group(1)

        return any(
            self._point_cloud_has_xyz_data(
                candidate
            )
            for candidate
            in pc_dir.glob(
                f"{prefix}_PC_*.xyz"
            )
        )

    @staticmethod
    def _sample_evenly(
        paths,
        count,
    ):
        if len(paths) <= count:
            return list(paths)

        indices = np.linspace(
            0,
            len(paths) - 1,
            num=count,
            dtype=int,
        )

        return [
            paths[int(index)]
            for index in indices
        ]

    def _build_autoplay_pool(self):
        """
        Build a small balanced demo pool without scanning every point cloud.

        Point-cloud availability is resolved only when each sample is loaded.
        This keeps Start demo responsive; samples without a PC automatically
        use the existing image-only fallback.
        """
        paths = self._get_all_acquisition_paths()

        frontal = []
        sideways = []

        for path in paths:
            orientation = self._orientation_for_path(
                path
            )

            if orientation == "sideways":
                sideways.append(path)
            else:
                frontal.append(path)

        frontal = self._sample_evenly(
            frontal,
            5,
        )

        sideways = self._sample_evenly(
            sideways,
            5,
        )

        pool = []

        for index in range(
            max(
                len(frontal),
                len(sideways),
            )
        ):
            if index < len(frontal):
                pool.append(
                    frontal[index]
                )

            if index < len(sideways):
                pool.append(
                    sideways[index]
                )

        return pool

    def toggle_autoplay(self):
        if self.autoplay_active:
            self._stop_autoplay()
            return

        if self.predictor is None:
            messagebox.showwarning(
                "Demo mode",
                "Load the multimodal model before starting demo mode.",
            )
            return

        self.status.set(
            "Preparing demo samples..."
        )
        self.update_idletasks()

        self.autoplay_pool = (
            self._build_autoplay_pool()
        )

        if not self.autoplay_pool:
            messagebox.showwarning(
                "Demo mode",
                "No acquisitions were found for demo mode.",
            )
            return

        self.autoplay_active = True
        self.autoplay_index = 0

        self.autoplay_button.configure(
            text="Stop demo"
        )

        self.status.set(
            "Demo mode started."
        )

        # Run the first sample on the next Tk event-loop turn so the
        # interface remains responsive and the button/status repaint first.
        self.autoplay_after_id = self.after(
            100,
            self._autoplay_step,
        )

    def _autoplay_step(self):
        if not self.autoplay_active:
            return

        # The callback that brought us here has already fired.
        self.autoplay_after_id = None

        try:
            path = self.autoplay_pool[
                self.autoplay_index
                % len(self.autoplay_pool)
            ]

            self.autoplay_index += 1

            self._load_image_path(
                path,
                source="Demo",
            )

            if self.predictor is not None:
                self.update_idletasks()
                self.run_prediction()

        except Exception as exc:
            self._stop_autoplay()

            self.status.set(
                f"Demo mode stopped: {exc}"
            )

            return

        if self.autoplay_active:
            self.autoplay_after_id = self.after(
                self.autoplay_interval_ms,
                self._autoplay_step,
            )

    def _stop_autoplay(self):
        if self.autoplay_after_id is not None:
            try:
                self.after_cancel(
                    self.autoplay_after_id
                )
            except Exception:
                pass

        self.autoplay_after_id = None
        self.autoplay_active = False

        if hasattr(
            self,
            "autoplay_button",
        ):
            self.autoplay_button.configure(
                text="Start demo"
            )

    def _mask_outline_changed(self):
        self._rebuild_view_a_display()

    @staticmethod
    def _apply_mask_outline(
        image: Image.Image,
        mask: Image.Image | None,
    ) -> Image.Image:
        if mask is None:
            return image

        mask_l = mask.convert("L")

        if mask_l.size != image.size:
            mask_l = mask_l.resize(
                image.size,
                Image.Resampling.NEAREST,
            )

        mask_array = np.asarray(
            mask_l
        )

        binary = (
            mask_array > 127
        )

        # Dataset masks may use either foreground polarity.
        if binary.mean() > 0.5:
            binary = ~binary

        binary_image = Image.fromarray(
            (
                binary.astype(
                    np.uint8
                )
                * 255
            )
        )

        eroded = np.asarray(
            binary_image.filter(
                ImageFilter.MinFilter(3)
            )
        ) > 0

        contour = (
            binary
            & ~eroded
        )

        rgba = np.asarray(
            image.convert("RGBA")
        ).copy()

        # Thin warm-yellow outline: clearly a visualization aid, not a model input.
        rgba[
            contour
        ] = np.array(
            [255, 215, 0, 255],
            dtype=np.uint8,
        )

        return Image.fromarray(
            rgba
        ).convert("RGB")

    def _rebuild_view_a_display(self):
        """
        Rebuild both RGB display panels.

        View A keeps the H/W or H/T measurement annotation.
        View B stays clean, except that the optional segmentation outline
        is drawn on both views when enabled.
        """
        if (
            self.view_a_image is None
            or self.view_b_image is None
        ):
            return

        # View A: model-result annotation.
        if self.last_prediction is not None:
            display_a = annotate_measurements(
                self.view_a_image,
                self.last_prediction,
                self.mask_a_image,
            )
        else:
            display_a = self.view_a_image.copy()

        # View B: clean RGB input.
        display_b = self.view_b_image.copy()

        if self.show_mask_outline.get():
            display_a = self._apply_mask_outline(
                display_a,
                self.mask_a_image,
            )

            display_b = self._apply_mask_outline(
                display_b,
                self.mask_b_image,
            )

        self.view_a_display = display_a
        self.view_b_display = display_b

        # Save annotated image still refers to View A.
        self.display_image = display_a

        if hasattr(
            self,
            "canvas_a",
        ):
            self.refresh_view_canvases()

    def _record_prediction_history(
        self,
        prediction,
    ):
        """
        Append one successful prediction to the current-session history.

        All three predicted quantities are stored, even when the selected
        acquisition orientation makes only H/W or H/T directly observable
        in the main GUI. Ground truth is copied at prediction time so later
        navigation cannot alter an older history entry.
        """
        if prediction is None:
            return

        image_path = (
            Path(self.image_path)
            if self.image_path is not None
            else None
        )

        acquisition = (
            image_path.stem[:-2]
            if (
                image_path is not None
                and image_path.stem.endswith(
                    ("_A", "_B")
                )
            )
            else (
                image_path.stem
                if image_path is not None
                else f"prediction_{len(self.prediction_history) + 1}"
            )
        )

        strand_id = None

        if image_path is not None:
            match = re.match(
                r"(\d{4})_\d{4}_[AB]$",
                image_path.stem,
                re.IGNORECASE,
            )

            if match:
                strand_id = int(
                    match.group(1)
                )

        ground_truth = (
            dict(self.ground_truth)
            if self.ground_truth is not None
            else None
        )

        self.prediction_history.append(
            {
                "sequence": len(self.prediction_history) + 1,
                "acquisition": acquisition,
                "strand_id": strand_id,
                "orientation": self.orientation.get(),
                "model": self.checkpoint_name_for_history(),
                "prediction": {
                    "height_mm": float(
                        prediction["height_mm"]
                    ),
                    "width_mm": float(
                        prediction["width_mm"]
                    ),
                    "thickness_mm": float(
                        prediction["thickness_mm"]
                    ),
                },
                "ground_truth": ground_truth,
            }
        )

        self.history_count_text.set(
            "Predictions in this session: "
            f"{len(self.prediction_history)}"
        )

        if hasattr(
            self,
            "history_button",
        ):
            self.history_button.configure(
                state="normal"
            )

        if (
            self.history_window is not None
            and self.history_window.winfo_exists()
        ):
            self._redraw_prediction_history()

    def checkpoint_name_for_history(self):
        if self.predictor is None:
            return "unknown model"

        try:
            return self.predictor.checkpoint_path.name
        except Exception:
            return "loaded model"

    @staticmethod
    def _history_reference_value(
        variable: tk.StringVar,
    ):
        text = variable.get().strip()

        if not text:
            return None

        # Accept both decimal point and decimal comma.
        return float(
            text.replace(",", ".")
        )

    def _history_reference_values(self):
        try:
            return {
                "height": self._history_reference_value(
                    self.history_ref_height
                ),
                "width": self._history_reference_value(
                    self.history_ref_width
                ),
                "thickness": self._history_reference_value(
                    self.history_ref_thickness
                ),
            }
        except ValueError:
            messagebox.showerror(
                "Reference value",
                "Reference values must be numeric values in millimetres "
                "or left empty.",
                parent=self.history_window,
            )
            return None

    def show_prediction_history(self):
        """
        Show all predictions accumulated since this GUI instance was opened.

        Three plots are shown: height, width, and thickness. Each compares the
        prediction sequence with the corresponding strand ground truth and can
        include an optional user-defined horizontal reference value.
        """
        if not self.prediction_history:
            messagebox.showwarning(
                "Prediction history",
                "Run at least one prediction first.",
            )
            return

        if (
            self.history_window is not None
            and self.history_window.winfo_exists()
        ):
            self.history_window.lift()
            self.history_window.focus_force()
            self._redraw_prediction_history()
            return

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            messagebox.showerror(
                "Prediction history",
                "Matplotlib is required for the history plot.\n\n"
                "Install it with:\n"
                "pip install matplotlib",
            )
            return

        window = tk.Toplevel(self)
        self.history_window = window

        window.title(
            "Prediction history — Granulo-10k"
        )

        # Size the history window from the current screen instead of using a
        # fixed geometry. This is much more reliable with Windows DPI scaling.
        window.update_idletasks()

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        history_width = min(
            1500,
            max(
                980,
                int(screen_width * 0.78),
            ),
        )
        history_height = min(
            1000,
            max(
                680,
                int(screen_height * 0.82),
            ),
        )

        history_x = max(
            0,
            (screen_width - history_width) // 2,
        )
        history_y = max(
            0,
            (screen_height - history_height) // 2,
        )

        window.geometry(
            f"{history_width}x{history_height}"
            f"+{history_x}+{history_y}"
        )
        window.minsize(
            900,
            650,
        )

        def on_close():
            try:
                if self.history_figure is not None:
                    import matplotlib.pyplot as plt
                    plt.close(
                        self.history_figure
                    )
            except Exception:
                pass

            self.history_window = None
            self.history_figure = None
            self.history_canvas = None
            self.history_axes = None
            window.destroy()

        window.protocol(
            "WM_DELETE_WINDOW",
            on_close,
        )

        outer = ttk.Frame(
            window,
            padding=12,
        )
        outer.pack(
            fill="both",
            expand=True,
        )

        controls = ttk.Frame(
            outer,
        )
        controls.pack(
            fill="x",
            pady=(0, 8),
        )

        # Row 0: session count + H/W/T reference controls.
        ttk.Label(
            controls,
            textvariable=self.history_count_text,
            font=("Segoe UI", 10, "bold"),
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            padx=(0, 18),
            pady=(0, 6),
            sticky="w",
        )

        ttk.Label(
            controls,
            text="Reference H [mm]:",
        ).grid(
            row=0,
            column=2,
            padx=(0, 4),
            pady=(0, 6),
            sticky="e",
        )

        ttk.Entry(
            controls,
            textvariable=self.history_ref_height,
            width=9,
        ).grid(
            row=0,
            column=3,
            padx=(0, 12),
            pady=(0, 6),
        )

        ttk.Label(
            controls,
            text="W [mm]:",
        ).grid(
            row=0,
            column=4,
            padx=(0, 4),
            pady=(0, 6),
            sticky="e",
        )

        ttk.Entry(
            controls,
            textvariable=self.history_ref_width,
            width=9,
        ).grid(
            row=0,
            column=5,
            padx=(0, 12),
            pady=(0, 6),
        )

        ttk.Label(
            controls,
            text="T [mm]:",
        ).grid(
            row=0,
            column=6,
            padx=(0, 4),
            pady=(0, 6),
            sticky="e",
        )

        ttk.Entry(
            controls,
            textvariable=self.history_ref_thickness,
            width=9,
        ).grid(
            row=0,
            column=7,
            padx=(0, 12),
            pady=(0, 6),
        )

        ttk.Button(
            controls,
            text="Apply reference",
            command=self._redraw_prediction_history,
        ).grid(
            row=0,
            column=8,
            padx=(0, 6),
            pady=(0, 6),
        )

        ttk.Button(
            controls,
            text="Clear references",
            command=self._clear_history_references,
        ).grid(
            row=0,
            column=9,
            pady=(0, 6),
        )

        # Row 1: smoothing controls. Keeping these on a separate row prevents
        # the control bar from forcing the entire window wider than the screen.
        ttk.Checkbutton(
            controls,
            text="EMA smoothing",
            variable=self.history_smoothing_enabled,
            command=self._redraw_prediction_history,
        ).grid(
            row=1,
            column=2,
            columnspan=2,
            padx=(0, 10),
            sticky="w",
        )

        ttk.Label(
            controls,
            text="Span:",
        ).grid(
            row=1,
            column=4,
            padx=(0, 4),
            sticky="e",
        )

        smoothing_spinbox = ttk.Spinbox(
            controls,
            from_=1,
            to=100,
            textvariable=self.history_ema_span,
            width=5,
            command=self._redraw_prediction_history,
        )
        smoothing_spinbox.grid(
            row=1,
            column=5,
            sticky="w",
        )

        smoothing_spinbox.bind(
            "<Return>",
            lambda _event: self._redraw_prediction_history(),
        )
        smoothing_spinbox.bind(
            "<FocusOut>",
            lambda _event: self._redraw_prediction_history(),
        )

        ttk.Label(
            controls,
            text="Higher span = stronger smoothing",
            foreground="#666666",
        ).grid(
            row=1,
            column=6,
            columnspan=4,
            padx=(12, 0),
            sticky="w",
        )

        controls.columnconfigure(
            0,
            weight=1,
        )

        history_help_label = ttk.Label(
            outer,
            text=(
                "Each point corresponds to one successful prediction in "
                "chronological order. Repeated acquisitions are kept. "
                "EMA span controls smoothing strength."
            ),
            foreground="#555555",
            justify="left",
            wraplength=max(
                700,
                history_width - 70,
            ),
        )
        history_help_label.pack(
            fill="x",
            anchor="w",
            pady=(0, 5),
        )

        figure = Figure(
            figsize=(9.0, 6.6),
            dpi=100,
        )

        axes = figure.subplots(
            3,
            1,
            sharex=True,
        )

        canvas = FigureCanvasTkAgg(
            figure,
            master=outer,
        )

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
        )

        self.history_figure = figure
        self.history_axes = axes
        self.history_canvas = canvas

        self._redraw_prediction_history()

    def _clear_history_references(self):
        self.history_ref_height.set("")
        self.history_ref_width.set("")
        self.history_ref_thickness.set("")

        if (
            self.history_window is not None
            and self.history_window.winfo_exists()
        ):
            self._redraw_prediction_history()

    @staticmethod
    def _ema_same_length(
        values: np.ndarray,
        span: int,
    ) -> np.ndarray:
        """
        Exponential moving average with the same output length as the input.

        The smoothing factor follows the common span convention:

            alpha = 2 / (span + 1)

        The first finite sample initializes the EMA. NaN samples remain NaN in
        the plotted output, while the previous EMA state is preserved so a
        later valid sample can continue the filtered sequence.
        """
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if len(values) == 0:
            return values.copy()

        span = max(
            1,
            int(span),
        )

        if span <= 1:
            return values.copy()

        alpha = 2.0 / (
            float(span) + 1.0
        )

        result = np.full_like(
            values,
            np.nan,
            dtype=np.float64,
        )

        previous = None

        for index, value in enumerate(
            values
        ):
            if not np.isfinite(value):
                continue

            if previous is None:
                previous = float(value)
            else:
                previous = (
                    alpha * float(value)
                    + (1.0 - alpha) * previous
                )

            result[index] = previous

        return result

    def _history_smoothing_settings(self):
        if not self.history_smoothing_enabled.get():
            return False, 1

        try:
            span = int(
                self.history_ema_span.get()
            )
        except (tk.TclError, ValueError):
            span = 10

        span = max(
            1,
            min(
                span,
                100,
            ),
        )

        try:
            current_span = int(
                self.history_ema_span.get()
            )
        except (tk.TclError, ValueError):
            current_span = None

        if current_span != span:
            self.history_ema_span.set(
                span
            )

        return True, span

    def _redraw_prediction_history(self):
        if (
            self.history_figure is None
            or self.history_axes is None
            or self.history_canvas is None
        ):
            return

        references = self._history_reference_values()

        if references is None:
            return

        smoothing_enabled, ema_span = (
            self._history_smoothing_settings()
        )

        history = self.prediction_history
        count = len(
            history
        )

        if count == 0:
            return

        x = np.arange(
            1,
            count + 1,
            dtype=np.int32,
        )

        task_specs = (
            (
                "height",
                "Height",
                "height_mm",
            ),
            (
                "width",
                "Width",
                "width_mm",
            ),
            (
                "thickness",
                "Thickness",
                "thickness_mm",
            ),
        )

        acquisition_labels = [
            item["acquisition"]
            for item in history
        ]

        for ax, (
            task_key,
            title,
            value_key,
        ) in zip(
            self.history_axes,
            task_specs,
        ):
            ax.clear()

            predicted = np.asarray(
                [
                    item["prediction"][
                        value_key
                    ]
                    for item in history
                ],
                dtype=np.float64,
            )

            ground_truth = np.asarray(
                [
                    (
                        item["ground_truth"][
                            value_key
                        ]
                        if (
                            item["ground_truth"]
                            is not None
                            and value_key
                            in item["ground_truth"]
                        )
                        else np.nan
                    )
                    for item in history
                ],
                dtype=np.float64,
            )

            if smoothing_enabled:
                predicted_plot = (
                    self._ema_same_length(
                        predicted,
                        ema_span,
                    )
                )
                ground_truth_plot = (
                    self._ema_same_length(
                        ground_truth,
                        ema_span,
                    )
                )

                prediction_label = (
                    f"Prediction (EMA span {ema_span})"
                )
                ground_truth_label = (
                    f"Ground truth (EMA span {ema_span})"
                )
            else:
                predicted_plot = predicted
                ground_truth_plot = ground_truth
                prediction_label = "Prediction"
                ground_truth_label = "Ground truth"

            ax.plot(
                x,
                predicted_plot,
                marker="o",
                markersize=4,
                linewidth=1.3,
                label=prediction_label,
            )

            if np.any(
                np.isfinite(
                    ground_truth_plot
                )
            ):
                ax.plot(
                    x,
                    ground_truth_plot,
                    marker="x",
                    markersize=5,
                    linewidth=1.1,
                    linestyle="--",
                    label=ground_truth_label,
                )

            reference = references[
                task_key
            ]

            if reference is not None:
                ax.axhline(
                    reference,
                    linestyle=":",
                    linewidth=1.4,
                    label=(
                        f"Reference = "
                        f"{reference:g} mm"
                    ),
                )

            ax.set_ylabel(
                f"{title} [mm]"
            )
            ax.grid(
                True,
                alpha=0.25,
            )
            ax.legend(
                loc="best",
            )

        bottom_ax = self.history_axes[
            -1
        ]

        if count <= 24:
            bottom_ax.set_xticks(
                x
            )
            bottom_ax.set_xticklabels(
                acquisition_labels,
                rotation=45,
                ha="right",
                fontsize=8,
            )
            bottom_ax.set_xlabel(
                "Acquisition"
            )
        else:
            bottom_ax.set_xlabel(
                "Prediction sequence"
            )

        title = (
            "Prediction history — current GUI session"
        )

        if smoothing_enabled:
            alpha = 2.0 / (
                float(ema_span) + 1.0
            )

            title += (
                f" — EMA span {ema_span} "
                f"(α={alpha:.3f})"
            )

        self.history_figure.suptitle(
            title
        )

        self.history_figure.tight_layout(
            rect=(0.02, 0.02, 0.99, 0.95)
        )

        self.history_canvas.draw_idle()

    def show_gate_visualization(self):
        prediction = self.last_prediction

        if (
            prediction is None
            or not prediction.get(
                "gate_top5"
            )
        ):
            messagebox.showwarning(
                "MMoE gates",
                "Run a prediction first.",
            )
            return

        window = tk.Toplevel(self)
        window.title(
            "MMoE gate activity — Top 5 experts"
        )
        window.geometry(
            "900x430"
        )
        window.minsize(
            760,
            360,
        )

        container = ttk.Frame(
            window,
            padding=14,
        )
        container.pack(
            fill="both",
            expand=True,
        )

        task_titles = {
            "height": "Height gate",
            "width": "Width gate",
            "thickness": "Thickness gate",
        }

        gates = prediction[
            "gate_top5"
        ]

        for column, task in enumerate(
            (
                "height",
                "width",
                "thickness",
            )
        ):
            frame = ttk.LabelFrame(
                container,
                text=task_titles[task],
                padding=10,
            )

            frame.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(
                    0 if column == 0 else 6,
                    0 if column == 2 else 6,
                ),
            )

            container.columnconfigure(
                column,
                weight=1,
            )

            entries = gates.get(
                task,
                [],
            )

            for row, entry in enumerate(
                entries
            ):
                percent = (
                    float(entry["weight"])
                    * 100.0
                )

                ttk.Label(
                    frame,
                    text=(
                        f"Expert "
                        f"{entry['expert']:02d}"
                    ),
                ).grid(
                    row=row,
                    column=0,
                    sticky="w",
                    pady=6,
                )

                progress = ttk.Progressbar(
                    frame,
                    maximum=100.0,
                    value=percent,
                    length=130,
                )

                progress.grid(
                    row=row,
                    column=1,
                    sticky="ew",
                    padx=8,
                )

                ttk.Label(
                    frame,
                    text=f"{percent:.1f}%",
                ).grid(
                    row=row,
                    column=2,
                    sticky="e",
                )

            frame.columnconfigure(
                1,
                weight=1,
            )

        ttk.Label(
            container,
            text=(
                "Each task-specific gate distributes probability over "
                f"{self.predictor.num_experts} shared experts."
            ),
            foreground="#555555",
        ).grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(14, 0),
        )

    def _resize_rgb_row_for_images(self):
        """
        Reserve enough vertical space to show the two RGB views close to
        native size, while leaving the rest of the viewer to the point cloud.
        """
        if (
            self.view_a_image is None
            or self.view_b_image is None
            or not hasattr(self, "viewer")
        ):
            return

        native_height = max(
            self.view_a_image.height,
            self.view_b_image.height,
        )

        # Keep some room for the point-cloud preview and labels.
        available_window_height = max(
            self.winfo_height(),
            700,
        )

        max_rgb_height = max(
            320,
            available_window_height - 360,
        )

        requested_height = int(
            min(
                native_height + 20,
                max_rgb_height,
            )
        )

        self.viewer.rowconfigure(
            1,
            minsize=requested_height,
            weight=0,
        )

        self.canvas_a.configure(
            height=requested_height,
        )

        self.canvas_b.configure(
            height=requested_height,
        )

    def _update_paired_views_label(self):
        try:
            image_a, image_b = (
                resolve_view_pair(
                    self.image_path
                )
            )

            self.input_pair_name.set(
                "Paired views: "
                f"{image_a.name} + {image_b.name}"
            )

        except Exception:
            self.input_pair_name.set(
                "Paired views: not available"
            )

    def _orientation_changed(self):
        self._update_visible_measurement_rows()
        self._select_point_cloud_for_orientation()
        self._update_input_mode_badge()

        # Manual override is allowed; an old prediction is no longer valid.
        self._reset_prediction_values()
        self._update_ground_truth_display()
        self._rebuild_view_a_display()

        self._update_point_cloud_preview()
        self._refresh_point_cloud_geometry()

    def _reset_prediction_values(self):
        self.last_prediction = None

        self.height_value.set("—")
        self.width_value.set("—")
        self.thickness_value.set("—")

        self.error_height_value.set("—")
        self.error_width_value.set("—")
        self.error_thickness_value.set("—")

        self.inference_time_text.set(
            "Inference: —"
        )

        if hasattr(
            self,
            "gates_button",
        ):
            self.gates_button.configure(
                state="disabled"
            )

    def run_prediction(self):
        if self.predictor is None:
            messagebox.showwarning(
                "No model",
                "Load a multimodal .pt checkpoint first.",
            )
            return

        if self.image_path is None:
            messagebox.showwarning(
                "No image",
                "Select a strand image first.",
            )
            return

        try:
            if self.point_cloud_path is None:
                self.status.set(
                    "Running image-only fallback "
                    f"({self.predictor.backbone_label} View A + View B)..."
                )
            else:
                self.status.set(
                    f"Running {self.predictor.backbone_label} + "
                    "PointNet++ + MMoE inference..."
                )

            self.update_idletasks()

            if self.predictor.device.type == "cuda":
                torch.cuda.synchronize()

            inference_start = time.perf_counter()

            prediction = self.predictor.predict(
                selected_image=self.image_path,
                point_cloud_path=self.point_cloud_path,
                orientation=self.orientation.get(),
            )

            if self.predictor.device.type == "cuda":
                torch.cuda.synchronize()

            inference_ms = (
                time.perf_counter()
                - inference_start
            ) * 1000.0

            self.inference_time_text.set(
                f"Inference: {inference_ms:.0f} ms"
            )

            self.last_prediction = prediction

            self.height_value.set(
                f"{prediction['height_mm']:.2f} mm"
            )

            self.width_value.set(
                f"{prediction['width_mm']:.2f} mm"
            )

            self.thickness_value.set(
                f"{prediction['thickness_mm']:.3f} mm"
            )

            # Ensure ground truth corresponds to the currently displayed
            # acquisition before calculating signed prediction errors.
            if self.ground_truth is None:
                self._load_ground_truth()
            else:
                self._update_ground_truth_display()

            self._update_error_display(
                prediction
            )

            self._record_prediction_history(
                prediction
            )

            self._rebuild_view_a_display()

            self.gates_button.configure(
                state="normal"
            )

            self._update_input_mode_badge(
                prediction.get("input_mode")
            )

            if prediction.get("input_mode") == "image_only":
                self.status.set(
                    "Image-only fallback prediction complete "
                    "(point cloud unavailable)."
                )
            else:
                self.status.set(
                    "Multimodal prediction complete."
                )

        except Exception as exc:
            messagebox.showerror(
                "Inference error",
                str(exc),
            )

            self.status.set(
                "Prediction failed."
            )

    def _update_error_display(
        self,
        prediction=None,
    ):
        if (
            prediction is None
            or self.ground_truth is None
        ):
            self.error_height_value.set("—")
            self.error_width_value.set("—")
            self.error_thickness_value.set("—")
            return

        delta_h = (
            prediction["height_mm"]
            - self.ground_truth["height_mm"]
        )

        delta_w = (
            prediction["width_mm"]
            - self.ground_truth["width_mm"]
        )

        delta_t = (
            prediction["thickness_mm"]
            - self.ground_truth["thickness_mm"]
        )

        self.error_height_value.set(
            f"{delta_h:+.2f} mm"
        )

        self.error_width_value.set(
            f"{delta_w:+.2f} mm"
        )

        self.error_thickness_value.set(
            f"{delta_t:+.3f} mm"
        )

    @staticmethod
    def _draw_image_on_canvas(
        canvas,
        image,
        empty_text,
    ):
        canvas.delete("all")

        canvas_width = max(
            canvas.winfo_width(),
            1,
        )
        canvas_height = max(
            canvas.winfo_height(),
            1,
        )

        if image is None:
            canvas.create_text(
                canvas_width / 2,
                canvas_height / 2,
                text=empty_text,
                fill="#c7c7c7",
                font=("Segoe UI", 15),
            )
            return None

        display = image.copy()

        display.thumbnail(
            (
                max(canvas_width - 24, 1),
                max(canvas_height - 24, 1),
            ),
            Image.Resampling.LANCZOS,
        )

        tk_image = ImageTk.PhotoImage(
            display
        )

        canvas.create_image(
            canvas_width / 2,
            canvas_height / 2,
            image=tk_image,
            anchor="center",
        )

        return tk_image

    def refresh_view_canvases(self):
        self.tk_image_a = self._draw_image_on_canvas(
            self.canvas_a,
            self.view_a_display,
            "View A",
        )

        self.tk_image_b = self._draw_image_on_canvas(
            self.canvas_b,
            self.view_b_display,
            "View B",
        )

    def _start_point_cloud_rotation(self):
        """
        Start slow automatic rotation of the embedded 3D point cloud.

        Rotation is driven by Tk's event loop, so it does not block the GUI.
        """
        self._stop_point_cloud_rotation()

        if (
            not PC_ROTATION_ENABLED
            or self.pc_ax is None
            or self.pc_figure_canvas is None
        ):
            return

        self.pc_rotation_after_id = self.after(
            PC_ROTATION_INTERVAL_MS,
            self._rotate_point_cloud_step,
        )

    def _rotate_point_cloud_step(self):
        """Advance the embedded point cloud by one azimuth step."""
        self.pc_rotation_after_id = None

        if (
            not PC_ROTATION_ENABLED
            or self.pc_ax is None
            or self.pc_figure_canvas is None
        ):
            return

        try:
            self.pc_rotation_angle = (
                self.pc_rotation_angle
                + PC_ROTATION_DEGREES_PER_STEP
            ) % 360.0

            self.pc_ax.view_init(
                elev=PC_ROTATION_ELEVATION,
                azim=self.pc_rotation_angle,
            )

            self.pc_figure_canvas.draw_idle()

            self.pc_rotation_after_id = self.after(
                PC_ROTATION_INTERVAL_MS,
                self._rotate_point_cloud_step,
            )

        except (tk.TclError, RuntimeError):
            # The canvas/window may have been destroyed while a scheduled
            # callback was pending.
            self.pc_rotation_after_id = None

    def _stop_point_cloud_rotation(self):
        """Cancel any scheduled embedded point-cloud rotation callback."""
        if self.pc_rotation_after_id is not None:
            try:
                self.after_cancel(
                    self.pc_rotation_after_id
                )
            except Exception:
                pass

        self.pc_rotation_after_id = None

    def _on_point_cloud_frame_resize(self, _event=None):
        """
        Keep the embedded Matplotlib figure synchronized with the actual
        Tk frame size. This is also triggered when the RGB row changes size.
        """
        if (
            self.pc_figure is None
            or self.pc_figure_canvas is None
        ):
            return

        width = max(
            self.pc_preview_frame.winfo_width(),
            1,
        )
        height = max(
            self.pc_preview_frame.winfo_height(),
            1,
        )

        # Ignore transient 1x1 geometry while Tk is rebuilding the layout.
        if width < 50 or height < 50:
            return

        try:
            dpi = float(
                self.pc_figure.get_dpi()
            )

            self.pc_figure.set_size_inches(
                width / dpi,
                height / dpi,
                forward=False,
            )

            widget = (
                self.pc_figure_canvas
                .get_tk_widget()
            )

            widget.configure(
                width=width,
                height=height,
            )

            self.pc_figure_canvas.draw_idle()

        except Exception:
            pass

    def _refresh_point_cloud_geometry(self):
        """
        Force Tk to finish geometry calculation, then resize/redraw the
        embedded point-cloud figure to the final available space.
        """
        self.update_idletasks()

        self._on_point_cloud_frame_resize()

        # One additional idle callback handles geometry changes caused by
        # newly loaded RGB images/captions.
        self.after_idle(
            self._on_point_cloud_frame_resize
        )

    def _clear_point_cloud_preview(self):
        self._stop_point_cloud_rotation()
        self.pc_ax = None

        if self.pc_figure_canvas is not None:
            try:
                self.pc_figure_canvas.get_tk_widget().destroy()
            except Exception:
                pass

        if self.pc_figure is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self.pc_figure)
            except Exception:
                pass

        self.pc_figure = None
        self.pc_figure_canvas = None

        if not self.pc_preview_placeholder.winfo_ismapped():
            self.pc_preview_placeholder.pack(
                fill="both",
                expand=True,
            )

    def _update_point_cloud_preview(self):
        self._clear_point_cloud_preview()

        if self.point_cloud_path is None:
            self.pc_preview_placeholder.configure(
                text=(
                    "No valid point cloud — prediction will use "
                    "View A + View B only"
                )
            )
            return

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            points = self._load_xyz_file(
                self.point_cloud_path
            )

            max_points = 12000

            if len(points) > max_points:
                step = max(
                    1,
                    len(points) // max_points,
                )
                display_points = points[::step]
            else:
                display_points = points

            self.pc_preview_placeholder.pack_forget()

            fig = Figure(
                figsize=(6.5, 2.7),
                dpi=100,
            )

            ax = fig.add_subplot(
                111,
                projection="3d",
            )

            self.pc_rotation_angle = PC_ROTATION_START_AZIMUTH
            ax.view_init(
                elev=PC_ROTATION_ELEVATION,
                azim=self.pc_rotation_angle,
            )

            ax.scatter(
                display_points[:, 0],
                display_points[:, 1],
                display_points[:, 2],
                s=0.7,
            )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.set_title(
                f"PointNet++ input — {len(points):,} points"
            )

            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            center = (mins + maxs) / 2.0
            radius = float(max(maxs - mins)) / 2.0

            if radius > 0:
                ax.set_xlim(
                    center[0] - radius,
                    center[0] + radius,
                )
                ax.set_ylim(
                    center[1] - radius,
                    center[1] + radius,
                )
                ax.set_zlim(
                    center[2] - radius,
                    center[2] + radius,
                )

            fig.tight_layout()

            canvas = FigureCanvasTkAgg(
                fig,
                master=self.pc_preview_frame,
            )

            canvas.draw()

            canvas.get_tk_widget().pack(
                fill="both",
                expand=True,
            )

            self.pc_figure = fig
            self.pc_figure_canvas = canvas
            self.pc_ax = ax

            self._start_point_cloud_rotation()

            # Synchronize the Matplotlib canvas with the space currently
            # available below the full-size RGB views.
            self._refresh_point_cloud_geometry()

            # Repaint RGB views after the embedded Matplotlib canvas has
            # participated in Tk's geometry calculation.
            self.after_idle(
                self.refresh_view_canvases
            )

        except Exception as exc:
            self.pc_preview_placeholder.configure(
                text=f"Point-cloud preview unavailable: {exc}"
            )

    def _update_mask_button(self):
        available = (
            self.mask_a_image is not None
            or self.mask_b_image is not None
        )

        self.mask_button.configure(
            state=(
                "normal"
                if available
                else "disabled"
            )
        )

    def open_segmentation_mask(self):
        if (
            self.mask_a_image is None
            and self.mask_b_image is None
        ):
            messagebox.showwarning(
                "No segmentation mask",
                "No segmentation mask is available for this acquisition.",
            )
            return

        window = tk.Toplevel(self)
        window.title("Segmentation masks")
        window.geometry("900x520")
        window.minsize(650, 400)

        container = ttk.Frame(
            window,
            padding=12,
        )
        container.pack(
            fill="both",
            expand=True,
        )

        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)

        ttk.Label(
            container,
            text="View A mask",
            font=("Segoe UI", 11, "bold"),
        ).grid(
            row=0,
            column=0,
            pady=(0, 6),
        )

        ttk.Label(
            container,
            text="View B mask",
            font=("Segoe UI", 11, "bold"),
        ).grid(
            row=0,
            column=1,
            pady=(0, 6),
        )

        canvas_a = tk.Canvas(
            container,
            background="#202124",
            highlightthickness=0,
        )
        canvas_a.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 6),
        )

        canvas_b = tk.Canvas(
            container,
            background="#202124",
            highlightthickness=0,
        )
        canvas_b.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(6, 0),
        )

        refs = {
            "a": None,
            "b": None,
        }

        def redraw(_event=None):
            refs["a"] = self._draw_image_on_canvas(
                canvas_a,
                (
                    self.mask_a_image.convert("RGB")
                    if self.mask_a_image is not None
                    else None
                ),
                "Mask not available",
            )

            refs["b"] = self._draw_image_on_canvas(
                canvas_b,
                (
                    self.mask_b_image.convert("RGB")
                    if self.mask_b_image is not None
                    else None
                ),
                "Mask not available",
            )

            window._mask_image_refs = refs

        canvas_a.bind(
            "<Configure>",
            redraw,
        )
        canvas_b.bind(
            "<Configure>",
            redraw,
        )

        redraw()


    def save_image(self):
        if self.last_prediction is None:
            messagebox.showwarning(
                "Nothing to save",
                "Run a prediction first.",
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Save annotated image",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
            ],
        )

        if filename:
            self.display_image.save(
                filename
            )

            self.status.set(
                f"Saved: {filename}"
            )

    @staticmethod
    def _point_cloud_has_xyz_data(
        path: Path,
    ) -> bool:
        if (
            not path.is_file()
            or path.stat().st_size == 0
        ):
            return False

        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                for line in f:
                    stripped = line.strip()

                    if (
                        not stripped
                        or stripped.startswith("#")
                    ):
                        continue

                    parts = (
                        stripped.replace(",", " ")
                        .split()
                    )

                    if len(parts) < 3:
                        continue

                    try:
                        xyz = np.asarray(
                            [
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                            ],
                            dtype=np.float32,
                        )
                    except ValueError:
                        continue

                    if np.all(
                        np.isfinite(xyz)
                    ):
                        return True

        except OSError:
            return False

        return False

    def _find_point_cloud(self):
        self.point_cloud_path = None
        self.point_cloud_candidates = []

        self.point_cloud_name.set(
            "Point cloud — not available"
        )

        self.point_cloud_button.configure(
            state="disabled"
        )

        if self.image_path is None:
            return

        match = re.match(
            r"(\d{4}_\d{4})_[AB]$",
            self.image_path.stem,
            re.IGNORECASE,
        )

        if not match:
            return

        acquisition_prefix = (
            match.group(1)
        )

        parts = list(
            self.image_path.parent.parts
        )

        try:
            images_index = parts.index(
                "Images"
            )
        except ValueError:
            return

        parts[images_index] = "PCs"

        pc_dir = Path(
            *parts
        )

        if not pc_dir.exists():
            return

        self.point_cloud_candidates = sorted(
            path
            for path in pc_dir.glob(
                f"{acquisition_prefix}_PC_*.xyz"
            )
            if self._point_cloud_has_xyz_data(
                path
            )
        )

        self._select_point_cloud_for_orientation()

    def _select_point_cloud_for_orientation(
        self,
    ):
        self.point_cloud_path = None

        if not self.point_cloud_candidates:
            self.point_cloud_name.set(
                "Point cloud — not available"
            )

            if hasattr(
                self,
                "point_cloud_button",
            ):
                self.point_cloud_button.configure(
                    state="disabled"
                )

            if hasattr(
                self,
                "pc_preview_frame",
            ):
                self._update_point_cloud_preview()

            if hasattr(
                self,
                "input_mode_badge",
            ):
                self._update_input_mode_badge()

            return

        orientation = self.orientation.get()

        if orientation == "sideways":
            preferred = [
                path
                for path
                in self.point_cloud_candidates
                if (
                    "thickness"
                    in path.stem.lower()
                    or "spess"
                    in path.stem.lower()
                )
            ]
        else:
            preferred = [
                path
                for path
                in self.point_cloud_candidates
                if (
                    "lungh"
                    in path.stem.lower()
                    or "largh"
                    in path.stem.lower()
                    or "width"
                    in path.stem.lower()
                    or "height"
                    in path.stem.lower()
                )
            ]

            if not preferred:
                preferred = [
                    path
                    for path
                    in self.point_cloud_candidates
                    if (
                        "thickness"
                        not in path.stem.lower()
                        and "spess"
                        not in path.stem.lower()
                    )
                ]

        self.point_cloud_path = (
            preferred[0]
            if preferred
            else self.point_cloud_candidates[0]
        )

        self.point_cloud_name.set(
            "Point cloud — "
            f"{self.point_cloud_path.name}"
        )

        if hasattr(
            self,
            "point_cloud_button",
        ):
            self.point_cloud_button.configure(
                state="normal"
            )

        if hasattr(
            self,
            "pc_preview_frame",
        ):
            self._update_point_cloud_preview()

        if hasattr(
            self,
            "input_mode_badge",
        ):
            self._update_input_mode_badge()

    @staticmethod
    def _load_xyz_file(
        path: Path,
    ) -> np.ndarray:
        rows = []

        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            for line in f:
                stripped = line.strip()

                if (
                    not stripped
                    or stripped.startswith("#")
                ):
                    continue

                parts = (
                    stripped.replace(",", " ")
                    .split()
                )

                if len(parts) < 3:
                    continue

                try:
                    xyz = [
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                    ]
                except ValueError:
                    continue

                if np.all(
                    np.isfinite(xyz)
                ):
                    rows.append(xyz)

        if not rows:
            raise ValueError(
                "The point-cloud file contains "
                "no valid XYZ points."
            )

        return np.asarray(
            rows,
            dtype=np.float32,
        )

    def view_point_cloud(self):
        if self.point_cloud_path is None:
            messagebox.showwarning(
                "No point cloud",
                "No matching valid point cloud "
                "was found for this acquisition.",
            )
            return

        try:
            import matplotlib.pyplot as plt

            points = self._load_xyz_file(
                self.point_cloud_path
            )

            max_points = 30000

            if len(points) > max_points:
                step = max(
                    1,
                    len(points) // max_points,
                )

                display_points = points[::step]
            else:
                display_points = points

            fig = plt.figure(
                figsize=(9, 7)
            )

            ax = fig.add_subplot(
                111,
                projection="3d",
            )

            ax.scatter(
                display_points[:, 0],
                display_points[:, 1],
                display_points[:, 2],
                s=1,
            )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            orientation_name = (
                "Frontal"
                if self.orientation.get() == "frontal"
                else "Sideways"
            )

            ax.set_title(
                f"{self.point_cloud_path.name}\n"
                f"{orientation_name} acquisition — "
                f"{len(points):,} points"
            )

            mins = points.min(
                axis=0
            )

            maxs = points.max(
                axis=0
            )

            center = (
                mins + maxs
            ) / 2.0

            radius = max(
                maxs - mins
            ) / 2.0

            if radius > 0:
                ax.set_xlim(
                    center[0] - radius,
                    center[0] + radius,
                )

                ax.set_ylim(
                    center[1] - radius,
                    center[1] + radius,
                )

                ax.set_zlim(
                    center[2] - radius,
                    center[2] + radius,
                )

            fig.tight_layout()
            plt.show(
                block=False
            )

            self.status.set(
                "Point cloud opened: "
                f"{self.point_cloud_path.name}"
            )

        except ImportError:
            messagebox.showerror(
                "Point-cloud viewer",
                "Matplotlib is required to display "
                "point clouds.\n\n"
                "Install it with:\n"
                "pip install matplotlib",
            )

        except Exception as exc:
            messagebox.showerror(
                "Point-cloud error",
                str(exc),
            )

    @staticmethod
    def _mask_path_for_image(
        image_path: Path,
    ) -> Path | None:
        parts = list(
            image_path.parts
        )

        try:
            images_index = parts.index(
                "Images"
            )
        except ValueError:
            return None

        parts[images_index] = "Masks"
        return Path(*parts)

    def _load_masks_for_views(
        self,
        image_a_path: Path,
        image_b_path: Path,
    ):
        self.mask_a_image = None
        self.mask_b_image = None
        self.mask_path = None
        self.mask_image = None

        for key, image_path in (
            ("a", image_a_path),
            ("b", image_b_path),
        ):
            candidate = self._mask_path_for_image(
                image_path
            )

            if (
                candidate is None
                or not candidate.exists()
            ):
                continue

            try:
                with Image.open(
                    candidate
                ) as mask:
                    loaded = mask.convert(
                        "L"
                    ).copy()

                if key == "a":
                    self.mask_a_image = loaded
                    self.mask_path = candidate
                else:
                    self.mask_b_image = loaded

            except Exception:
                continue

        self.mask_image = self.mask_a_image

    def _load_mask(self):
        self.mask_path = None
        self.mask_image = None

        if self.image_path is None:
            return

        parts = list(
            self.image_path.parts
        )

        try:
            images_index = parts.index(
                "Images"
            )
        except ValueError:
            return

        parts[images_index] = "Masks"

        candidate = Path(
            *parts
        )

        if not candidate.exists():
            return

        try:
            with Image.open(
                candidate
            ) as mask:
                self.mask_image = (
                    mask.convert("L").copy()
                )

            self.mask_path = candidate

        except Exception:
            self.mask_path = None
            self.mask_image = None

    def _candidate_measurements_files(
        self,
        image_path: Path,
    ) -> list[Path]:
        """
        Return plausible measurements.txt locations, nearest/most specific first.

        Standard Granulo-10k layout:
            Images/
                Strands_compliant/
                    measurements.txt
                    001/
                        0001_0001_A.png

        The fallback ancestor search also keeps the GUI working if the dataset
        is moved or a slightly different local layout is used.
        """
        candidates = []

        # Prefer the canonical Strands_compliant directory explicitly.
        for parent in (
            image_path.parent,
            *image_path.parents,
        ):
            if parent.name == "Strands_compliant":
                candidate = parent / "measurements.txt"

                if candidate.exists():
                    candidates.append(
                        candidate
                    )

                break

        # General ancestor fallback.
        current = image_path.parent

        for _ in range(10):
            candidate = (
                current
                / "measurements.txt"
            )

            if (
                candidate.exists()
                and candidate not in candidates
            ):
                candidates.append(
                    candidate
                )

            if current.parent == current:
                break

            current = current.parent

        return candidates

    def _load_measurements_cached(
        self,
        measurements_file: Path,
    ):
        key = str(
            measurements_file.resolve()
        )

        if key not in self._measurements_cache:
            self._measurements_cache[key] = (
                load_measurements(
                    measurements_file
                )
            )

        return self._measurements_cache[key]

    def _load_ground_truth(self):
        self.ground_truth = None

        self.gt_height_value.set("—")
        self.gt_width_value.set("—")
        self.gt_thickness_value.set("—")

        if self.image_path is None:
            return

        match = re.match(
            r"(\d{4})_\d{4}_[AB]$",
            self.image_path.stem,
            re.IGNORECASE,
        )

        if not match:
            return

        strand_id = int(
            match.group(1)
        )

        # Try every plausible measurements file instead of stopping at the
        # first file that merely exists.
        for measurements_file in self._candidate_measurements_files(
            self.image_path
        ):
            try:
                measurements = (
                    self._load_measurements_cached(
                        measurements_file
                    )
                )
            except Exception:
                continue

            ground_truth = measurements.get(
                strand_id
            )

            if ground_truth is not None:
                self.ground_truth = ground_truth
                break

        self._update_ground_truth_display()


    def _update_ground_truth_display(self):
        if self.ground_truth is None:
            self.gt_height_value.set("—")
            self.gt_width_value.set("—")
            self.gt_thickness_value.set("—")
            return

        self.gt_height_value.set(
            f"{self.ground_truth['height_mm']:.2f} mm"
        )

        self.gt_width_value.set(
            f"{self.ground_truth['width_mm']:.2f} mm"
        )

        self.gt_thickness_value.set(
            f"{self.ground_truth['thickness_mm']:.3f} mm"
        )


def main():
    parser = argparse.ArgumentParser(
        description=APP_TITLE
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional path to a trained multimodal "
            ".pt checkpoint"
        ),
    )

    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Optional path to a strand image to load "
            "when the GUI starts"
        ),
    )

    args = parser.parse_args()

    StrandDemoGUI(
        checkpoint=args.model,
        image=args.image,
    ).mainloop()


if __name__ == "__main__":
    main()
