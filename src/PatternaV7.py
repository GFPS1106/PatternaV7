"""
Patterna V7
Shape-agnostic spatial quantification of micropatterned colonies.

Patterna is an image-analysis pipeline designed to quantify immunofluorescence
patterns in micropatterned pluripotent stem-cell colonies and 2D gastruloid-like
systems. The pipeline converts each colony into a normalized spatial coordinate
system using a spiderweb sampling strategy, where fluorescence intensities are
sampled from the colony centroid to the true colony boundary across many angular
directions.

Instead of relying on one or two manually selected line profiles, Patterna
captures marker organization across the full micropattern geometry. This allows
radial position to be defined consistently from center to edge, independent of
whether the colony is circular, oval, triangular, tear-shaped, or irregular.

## Main analysis features:

1. Colony masking and quality-control overlays
2. Shape-agnostic spiderweb sampling of each colony
3. Channel x radius x angle intensity grids
4. Radial intensity profiles for each marker
5. Per-channel max-normalized radial profiles for curve-shape comparison
6. Angle-radius rectangular topography maps
7. Polar topography visualizations
8. Composite rectangular and polar topography maps
9. Orthogonal major/minor axis profiles and anisotropy metrics
10. Pattern fingerprints and spatial summary metrics
11. Grouped radial profiles across multiple micropatterns
12. Grouped heatmaps and grouped polar topography maps
13. Grouped orthogonal profile plots
14. Single-cell/nuclei-level marker quantification
15. StarDist-based nuclei segmentation with optional hybrid watershed fallback
16. Cell phenotype classification based on configurable marker thresholds
17. Cell-intensity maps for marker expression visualization
18. Standalone cell phenotype maps
19. Integrated spatial summary panels combining marker maps, phenotype maps,
    profiles, and quantitative summaries

## Input:

* OME-TIFF or TIFF microscopy images
* Expected format: multi-channel image stacks containing DAPI and marker channels
* Current default channel mapping:
  clean ch0 = DAPI
  clean ch1 = TBXT
  clean ch2 = SOX2
  clean ch3 = GATA6
* If a 5-channel image is detected, the script removes the brightfield channel
  and keeps the cleaned fluorescence stack for analysis.

## Main output folders:

A Patterna output directory may contain:

* masks/
  Binary colony masks.

* grids/
  Spiderweb-sampled intensity grids saved as NumPy arrays.
  Shape: channel x radius x angle.

* profiles/
  Per-image radial profiles, orthogonal major/minor profiles, and profile CSVs.

* heatmaps/
  Per-image rectangular angle-radius topography maps and spiderweb heatmaps.

* topography_maps/
  Per-image polar topography maps and composite polar maps.

* debug/
  Raw channel QC images, cleaned channel QC images, and spiderweb overlays.

* qc/
  Nuclei segmentation QC overlays, including boundary and colored-instance maps.

* cells/
  Per-cell measurement tables and instance-label masks.

* metrics/
  Quantitative summary tables including spatial metrics, topography summaries,
  orthogonal metrics, pattern fingerprints, and combined cell tables.

* excel/
  Organized Excel workbooks with radial profiles grouped by channel.

* grouped_profiles/
  Mean ± variability radial profile plots across multiple micropatterns.

* grouped_heatmaps/
  Grouped rectangular heatmaps averaged across micropatterns.

* grouped_topography_maps/
  Grouped polar heatmaps averaged across micropatterns.

* grouped_orthogonal/
  Grouped major/minor axis profile plots.

* cell_phenotypes/
  Cell phenotype classification tables, threshold diagnostics, and phenotype
  summary plots.

* cell_intensity_maps/
  Per-marker cell-level intensity maps for each micropattern.

* cell_phenotype_maps/
  Standalone phenotype maps showing classified cells.

* grouped_cell_intensity_maps/
  Grouped disk-rendered marker maps based on averaged spiderweb grids.

* summary_panels/
  Integrated spatial summary panels combining cell maps, phenotype summaries,
  marker-positive fractions, radial profiles, and anisotropy information.

## Important notes:

* Patterna is intended for research-use image analysis and exploratory biological
  quantification.
* Marker-positive cell thresholds should be calibrated using appropriate controls
  before final biological interpretation.
* Percentile-based phenotype thresholding is useful for exploration and comparison,
  but should not be treated as final biological ground truth without validation.
* Nuclei segmentation parameters may need adjustment depending on colony size,
  image resolution, DAPI quality, and cell density.
* StarDist segmentation is used by default in this version; the hybrid mode can
  fall back to dense watershed segmentation if StarDist under-segments highly
  packed colonies.
  """

import os
import glob
from cv2.gapi import mask
import numpy as np
import pandas as pd
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy import ndimage as ndi
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from scipy import ndimage as ndi
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Optional StarDist nuclei segmentation
try:
    from stardist.models import StarDist2D
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False

# ============================================================
# Patterna visual palette
# Inspired by high-contrast developmental atlas palettes:
# separated blue / teal / violet / raspberry tones.
# ============================================================

PATTERNA_PALETTE = {
    0: "#2563EB",  # blue
    1: "#F97316",  # orange
    2: "#C026D3",  # magenta
    3: "#16A34A",  # green
}

PATTERNA_MAIN_PAIR = {
    "primary": "#2563EB",   # blue
    "secondary": "#F97316", # orange
}


def make_channel_cmap(hex_color, name):
    """
    Single-channel colormap: white -> channel color.
    Used for rectangular and polar topography maps.
    """
    return LinearSegmentedColormap.from_list(
        name,
        ["#FFFFFF", hex_color]
    )

PATTERNA_CMAPS = {
    ch: make_channel_cmap(color, f"patterna_ch{ch}_cmap")
    for ch, color in PATTERNA_PALETTE.items()
}

def get_channel_color(ch):
    return PATTERNA_PALETTE.get(int(ch), "#222222")

def get_channel_cmap(ch):
    return PATTERNA_CMAPS.get(int(ch), "viridis")
    
# ============================================================
# Phenotype-class palette
# Single-positive classes use the channel colors.
# Co-expression classes use companion colors that match Patterna.
# ============================================================

PHENOTYPE_CLASS_COLOR_OVERRIDES = {
    "negative": "#6B7280",           # neutral gray

    # single positives: direct Patterna channel colors
    "TBXT only": "#F97316",          # clean ch1
    "SOX2 only": "#C026D3",          # clean ch2
    "GATA6 only": "#16A34A",         # clean ch3

    # double/triple positives: companion colors
    "TBXT+SOX2": "#E11D48",          # raspberry/coral
    "TBXT+GATA6": "#84CC16",         # yellow-green
    "SOX2+GATA6": "#0891B2",         # teal
    "TBXT+SOX2+GATA6": "#7C3AED",    # violet
}


def marker_name_to_channel(marker_name, channel_names=None):
    names = CHANNEL_NAMES if channel_names is None else channel_names
    target = str(marker_name).replace(" only", "").strip()

    for ch, nm in names.items():
        if str(nm).strip() == target:
            return int(ch)

    return None


def get_marker_color(marker_name, channel_names=None):
    ch = marker_name_to_channel(marker_name, channel_names=channel_names)
    if ch is None:
        return "#444444"
    return get_channel_color(ch)


def get_phenotype_color(phenotype_class, channel_names=None):
    label = str(phenotype_class)

    if label in PHENOTYPE_CLASS_COLOR_OVERRIDES:
        return PHENOTYPE_CLASS_COLOR_OVERRIDES[label]

    if label == "negative":
        return PHENOTYPE_CLASS_COLOR_OVERRIDES["negative"]

    # fallback for unexpected phenotype names
    markers = [m.replace(" only", "").strip() for m in label.split("+") if m.strip()]

    if len(markers) == 1:
        return get_marker_color(markers[0], channel_names=channel_names)

    return "#444444"

# =========================
# CONFIG (VS Code friendly)
# =========================
RUN_WITH_CONFIG = True   # <- set False if you ever want CLI args

INPUT_DIR = r"C:\Users\grezc\Desktop\Patterna repository\test_data"
OUT_DIR   = r"C:\Users\grezc\Desktop\PatternaV7_test_for_repository"  # output folder on Desktop

#Z handling: "max", "mean", "mid", or int index (0-based)
Z_MODE = "max"

#Mask Settings

MASK_MODE     = "dapi"   # "composite" (best), "dapi" or "dapi_sauvola" (most sensitive, good for nuclei-rich channels)
DAPI_CHANNEL  = 0             # only used if MASK_MODE="dapi"
THRESH_METHOD = "percentile"        # "otsu" or "percentile"
THRESH_PCT    = 70   # only used if THRESH_METHOD="percentile"

N_ANGLES = 240
N_R      = 200
SIGMA_R  = 2.5

SAVE_HEATMAPS = True 
SAVE_DEBUG    = True    # overlay spokes/rings to sanity-check
SAVE_GRIDS    = True    # saves .npy for future biological metrics
SAVE_MASKS    = True    # saves mask per sample

# =============================
# Grouped profile export settings
# =============================
EXPORT_CHANNEL_PROFILE_WORKBOOK = True
EXPORT_GROUPED_HEATMAPS = True
EXPORT_GROUPED_ORTHOGONAL = True
EXPORT_INTEGRATED_SUMMARY_PANELS = True

EXPORT_STANDALONE_CELL_MAPS = True
EXPORT_GROUPED_DISK_MARKER_MAPS = True

# for per-cell intensity map rendering
CELL_INTENSITY_MAP_STAT = "mean"   # or "p90"
CELL_INTENSITY_MAP_BG = "white"    # "white" or "black"

# How cell-intensity maps are displayed.
# Options:
#   "per_image_percentile" = each MP gets its own display range; pretty but NOT comparable
#   "per_image_minmax"     = each MP min/max; exploratory only
#   "global_percentile"    = one display range per marker across all MPs in this run; best for comparison
#   "manual"              = use CELL_INTENSITY_MANUAL_LIMITS below
CELL_INTENSITY_NORM_MODE = "global_percentile"
CELL_INTENSITY_NORM_PCTS = (0.5, 99.5)

# Optional manual display limits for cell-intensity maps, in raw cell-intensity units.
# Only used when CELL_INTENSITY_NORM_MODE = "manual".
# Leave as None to calculate from the data.
CELL_INTENSITY_MANUAL_LIMITS = {
    1: None,  # TBXT, example: (0, 160)
    2: None,  # SOX2
    3: None,  # GATA6
}


# If True, Patterna will combine all MPs into one Excel workbook.
# Each channel gets its own sheet with all replicate profiles side by side.
GROUP_PROFILE_REPLICATES = True

# Variability shown in exported summary columns and grouped plots.
# Options: "sd" or "sem"
PROFILE_ERROR_MODE = "sd"

# Optional readable channel names.
# Adjust depending on your cocktail.
CHANNEL_NAMES = {
    0: "DAPI",
    1: "TBXT",
    2: "SOX2",
    3: "GATA6",
}

# =============================
# DELIVERABLE 2: single-cell (nuclei) settings
# =============================
DO_CELLS = True        # turn nuclei quantification on/off
NUC_DAPI_CHANNEL = 0     # which channel to segment nuclei from (usually DAPI)
N_SECTORS = 12           # number of angular bins for sector analysis
RING_EDGES = (0.33, 0.67)  # inner/mid/outer boundaries in r_frac

# nuclei segmentation QC filters (pixels)
MIN_NUC_AREA = 35
MAX_NUC_AREA = 300
# segmentation smoothing
NUC_GAUSS_SIGMA = 0.8

SAVE_CELL_OVERLAY = True  # save debug overlay of nuclei segmentation + sector/ring boundaries

# =============================
# Cell phenotype classification settings
# =============================
EXPORT_CELL_PHENOTYPES = True

# Channels to classify as biological markers.
# Usually skip DAPI because DAPI is nuclear counterstain, not a marker phenotype.
PHENOTYPE_MARKER_CHANNELS = [1, 2, 3]

# How to define marker-positive cells.
# Options:
#   "manual"               = use explicit thresholds from PHENOTYPE_MANUAL_THRESHOLDS
#   "control"              = calculate thresholds from selected control / negative-control images
#   "percentile_global"    = exploratory threshold from all cells pooled across images
#   "percentile_per_image" = exploratory threshold separately per image
PHENOTYPE_THRESHOLD_MODE = "percentile_global"

# Exploratory percentile used only for percentile_global / percentile_per_image modes.
# This is NOT final biology; calibrate with Thorsten/Makis using controls.
PHENOTYPE_POSITIVE_PERCENTILE = 75

# Optional marker-specific percentiles for exploratory thresholding.
# Use this when one marker is too permissive/strict under the shared percentile.
# Example: make GATA6 stricter with 90–95 while keeping SOX2 more permissive.
# Set values to None to fall back to PHENOTYPE_POSITIVE_PERCENTILE.
PHENOTYPE_POSITIVE_PERCENTILE_BY_CHANNEL = {
    1: 75,  # TBXT
    2: 60,  # SOX2
    3: 90,  # GATA6
}

# Manual thresholds, in the same units as the chosen cell intensity statistic
# (PHENOTYPE_INTENSITY_STAT: usually ch*_mean or ch*_p90).
# Use channel numbers as keys. Set to None if you do not want to use manual mode yet.
PHENOTYPE_MANUAL_THRESHOLDS = {
    1: None,  # TBXT
    2: None,  # SOX2
    3: None,  # GATA6
}

# Control-derived thresholds.
# The entries are matched as substrings inside the image names from the cell table.
# Example: ["CTRL"] will use any image whose name contains "CTRL".
# For proper negative controls, you can set marker-specific control images below.
CONTROL_IMAGES_FOR_THRESHOLDS = ["CTRL", "control"]

# Optional marker-specific control image substrings.
# If a channel list is non-empty, it overrides CONTROL_IMAGES_FOR_THRESHOLDS for that channel.
# This is useful if each marker has a different negative-control source.
CONTROL_IMAGES_FOR_THRESHOLDS_BY_CHANNEL = {
    1: [],  # TBXT negative-control image name substrings
    2: [],  # SOX2 negative-control image name substrings
    3: [],  # GATA6 negative-control image name substrings
}

# How to calculate the threshold from the selected control cells.
# Options: "mean_plus_2sd", "mean_plus_3sd", "percentile_95", "percentile_99"
CONTROL_THRESHOLD_METHOD = "mean_plus_2sd"

# Which cell intensity measurement to use.
# Options usually available from your cell table:
#   "mean" or "p90"
PHENOTYPE_INTENSITY_STAT = "mean"

# =============================
# Nuclei segmentation mode — Shapes V2 hybrid
# =============================
# Uses StarDist first. If StarDist severely under-segments dense 800 µm MPs,
# Patterna automatically switches to a dense-DAPI watershed fallback.
# There is NO Cellpose in this workflow because it is too slow for this batch.
NUC_SEGMENTATION_MODE = "stardist"  # "hybrid" or "stardist"

# StarDist first-pass settings
STARDIST_MODEL_NAME = "2D_versatile_fluo"
STARDIST_PROB_THRESH = 0.30
STARDIST_NMS_THRESH = 0.55
#STARDIST_PROB_THRESH = 0.30 AND STARDIST_NMS_THRESH = 0.55 is a good starting point for 500um not so dense colonies 
#Turn OFF area filtering (add None instead of any pixel number)
STARDIST_MIN_SIZE = None
STARDIST_MAX_SIZE = None
STARDIST_BBOX_PAD = 30

# Hybrid safeguard
# If StarDist finds fewer than this many nuclei, switch to dense watershed.
# For Shapes V2, 90–120 cells is biologically impossible, so 800 is a safe first guardrail.
HYBRID_MIN_ACCEPTABLE_NUCLEI = 800

# Dense watershed fallback settings for very packed Shapes V2 DAPI
DENSE_WS_MIN_AREA = 10
DENSE_WS_MAX_AREA = 500
DENSE_WS_GAUSS_SIGMA = 0.50
DENSE_WS_BG_SIGMA = 30
DENSE_WS_MIN_DISTANCE = 3
DENSE_WS_BLOCK_SIZE = 31
DENSE_WS_C = 0
# -------------------------
# Utilities
# -------------------------

#helper convert hex to RBG 
def hex_to_bgr01(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return np.array([b, g, r], dtype=np.float32)
#helper make 4 channel comppisute using Patterna pallete 
def make_patterna_composite(img_CYX, max_channels=4):
    """
    Build a BGR composite using the Patterna channel palette.
    Includes up to 4 channels.
    """
    C, H, W = img_CYX.shape
    out = np.zeros((H, W, 3), dtype=np.float32)

    for ch in range(min(C, max_channels)):
        img_norm = normalize01(img_CYX[ch])
        color = hex_to_bgr01(get_channel_color(ch))  # BGR in [0,1]
        out += img_norm[..., None] * color[None, None, :]

    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)

#compisite grid RGB rendering helper
def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float32)

# helper for composite grid RGB rendering (polar and rectangular)
def make_composite_grid_rgb(grid, max_channels=4):
    """
    Convert grid (C,R,A) into a composite RGB image (R,A,3)
    using the Patterna palette.
    """
    C, R, A = grid.shape
    rgb = np.zeros((R, A, 3), dtype=np.float32)

    for ch in range(min(C, max_channels)):
        color = hex_to_rgb01(get_channel_color(ch))   # RGB in [0,1]
        layer = grid[ch].astype(np.float32)           # (R,A), assumed already normalized 0-1
        rgb += layer[..., None] * color[None, None, :]

    rgb = np.clip(rgb, 0, 1)
    return rgb

#helper rectangular map overlay 
def save_composite_rectangular_map(stem, out_dir, grid, r_fracs):
    """
    Save one rectangular composite topography map with all channels merged.
    """
    rgb = make_composite_grid_rgb(grid)   # (R,A,3)

    plt.figure(figsize=(8, 4))
    plt.imshow(
        rgb,
        aspect="auto",
        origin="lower",
        extent=[0, 360, float(r_fracs[0]), float(r_fracs[-1])]
    )
    plt.title(f"{stem} — Composite topography map")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Radius fraction")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "heatmaps", f"{stem}_topography_composite_rect.png"),
        dpi=220
    )
    plt.close()

#helper polar map overlay (spiderweb grid rendered into circle)
def save_composite_polar_map(stem, out_dir, grid, size=800):
    """
    Save one composite polar-style map by rendering the spiderweb grid
    into a circular image.
    """
    rgb_grid = make_composite_grid_rgb(grid)   # (R,A,3)

    R, A, _ = rgb_grid.shape
    center = (size - 1) / 2.0
    radius_max = center

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = xx - center
    dy = yy - center

    rr = np.sqrt(dx**2 + dy**2) / radius_max          # normalized radius
    theta = np.arctan2(-dy, dx)                       # image coords -> angle
    theta[theta < 0] += 2 * np.pi

    a_idx = theta / (2 * np.pi) * (A - 1)
    r_idx = rr * (R - 1)

    inside = rr <= 1.0

    out = np.zeros((size, size, 3), dtype=np.float32)

    for c in range(3):
        sampled = map_coordinates(
            rgb_grid[:, :, c],
            [r_idx, a_idx],
            order=1,
            mode="nearest"
        )
        out[:, :, c] = sampled * inside

    out = np.clip(out, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(out, origin="upper")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "topography_maps", f"{stem}_topography_composite_polar.png"),
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.close()

# helper make compouste image for spiderweb 
def make_composite_image_bgr(img_CYX, max_channels=4):
    """
    Build a composite BGR image from all channels using the Patterna palette.
    Returns uint8 image (H,W,3) ready for OpenCV drawing/saving.
    """
    C, H, W = img_CYX.shape
    out = np.zeros((H, W, 3), dtype=np.float32)

    for ch in range(min(C, max_channels)):
        img_norm = normalize01(img_CYX[ch])
        color = hex_to_bgr01(get_channel_color(ch))   # BGR in [0,1]
        out += img_norm[..., None] * color[None, None, :]

    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)

# Orthogonal metrics 
# -------------------------
def principal_axis_angle(mask_u8):
    """Return principal axis angle theta (radians) of mask via PCA of pixel coords."""
    ys, xs = np.nonzero(mask_u8 > 0)
    if len(xs) < 10:
        return 0.0
    X = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    X -= X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / (X.shape[0] + 1e-8)
    w, v = np.linalg.eigh(cov)  # eigenvectors columns
    v_major = v[:, np.argmax(w)]
    theta = float(np.arctan2(v_major[1], v_major[0]))  # angle of major axis
    return theta

def angle_to_index(theta, n_angles):
    theta = float(theta) % (2*np.pi)
    return int(round(theta / (2*np.pi) * n_angles)) % n_angles

def sector_indices(center_idx, half_width, n_angles):
    # indices in [center-half_width, center+half_width] with wrap-around
    idxs = [(center_idx + k) % n_angles for k in range(-half_width, half_width+1)]
    return np.array(idxs, dtype=int)

def orthogonal_profiles_from_grid(grid_ch_RA, theta0, n_angles, half_width_bins=3):
    """
    grid_ch_RA: (R,A) for one channel
    returns p_major(R,), p_minor(R,) by averaging small angular sectors around the two directions.
    """
    a0  = angle_to_index(theta0, n_angles)
    a90 = angle_to_index(theta0 + np.pi/2, n_angles)

    # average both opposite directions to be symmetric
    maj_idxs = np.concatenate([
        sector_indices(a0, half_width_bins, n_angles),
        sector_indices((a0 + n_angles//2) % n_angles, half_width_bins, n_angles)
    ])
    min_idxs = np.concatenate([
        sector_indices(a90, half_width_bins, n_angles),
        sector_indices((a90 + n_angles//2) % n_angles, half_width_bins, n_angles)
    ])

    p_major = grid_ch_RA[:, maj_idxs].mean(axis=1)
    p_minor = grid_ch_RA[:, min_idxs].mean(axis=1)
    return p_major.astype(np.float32), p_minor.astype(np.float32)

def band_mean(profile, r_fracs, lo, hi):
    idx = np.where((r_fracs >= lo) & (r_fracs <= hi))[0]
    if len(idx) == 0:
        idx = np.array([np.argmin(np.abs(r_fracs - (lo+hi)/2))])
    return float(profile[idx].mean())

def anisotropy_index(pmaj, pmin, eps=1e-8):
    num = float(np.linalg.norm(pmaj - pmin))
    den = float(np.linalg.norm(pmaj) + np.linalg.norm(pmin) + eps)
    return num / den
#-------------------------
#------------------------- Deliverable 1: 3 ring fractions + moments (shape-agnostic) ortogonal ware metrics Helpers

def robust_bg_and_scale(radial, bg_q=0.10, hi_q=0.95, eps=1e-8):
    """Background-subtract + robust-scale a 1D radial curve."""
    bg = float(np.quantile(radial, bg_q))
    x = radial - bg
    x[x < 0] = 0
    hi = float(np.quantile(x, hi_q))
    if hi < eps:
        return x * 0.0, bg, hi
    return x / (hi + eps), bg, hi

def ring_fracs_from_mass(mass, r_fracs, cuts=(1/3, 2/3), eps=1e-8):
    """Given nonnegative mass vs r, return fractions in inner/mid/outer that sum to 1."""
    total = float(np.sum(mass))
    if total < eps:
        return 0.0, 0.0, 0.0

    c1, c2 = cuts
    inner = float(np.sum(mass[r_fracs < c1])) / total
    mid   = float(np.sum(mass[(r_fracs >= c1) & (r_fracs < c2)])) / total
    outer = float(np.sum(mass[r_fracs >= c2])) / total
    return inner, mid, outer

def r_moments_from_mass(mass, r_fracs, eps=1e-8):
    """Return r_mean, r_peak, r_width (std) from nonnegative mass vs r."""
    total = float(np.sum(mass))
    if total < eps:
        return 0.0, 0.0, 0.0
    p = mass / (total + eps)
    r_mean = float(np.sum(r_fracs * p))
    r_peak = float(r_fracs[int(np.argmax(p))])
    r_var  = float(np.sum(((r_fracs - r_mean) ** 2) * p))
    r_width = float(np.sqrt(max(r_var, 0.0)))
    return r_mean, r_peak, r_width
#----------------------------
#New helper function for updated individual channel sheets
def ensure_dirs(base_out):
    for sub in [
        "heatmaps",
        "profiles",
        "debug",
        "metrics",
        "grids",
        "masks",
        "cells",
        "qc",
        "topography_maps",
        "excel",
        "grouped_profiles",
        "grouped_heatmaps",
        "grouped_topography_maps",
        "grouped_orthogonal",
        "cell_phenotypes",
        "summary_panels",
        "cell_intensity_maps",
        "cell_phenotype_maps",
        "grouped_cell_intensity_maps",
    ]:
        os.makedirs(os.path.join(base_out, sub), exist_ok=True)
#helper for variability curve (sd or sem) across replicates for grouped profiles
def error_curve(values_2d, mode="sd"):
    """
    values_2d: array with shape (n_replicates, n_radii)
    Returns variability curve across replicates.
    """
    values_2d = np.asarray(values_2d, dtype=np.float32)

    if values_2d.ndim != 2 or values_2d.shape[0] == 0:
        return np.array([], dtype=np.float32)

    if values_2d.shape[0] == 1:
        return np.zeros(values_2d.shape[1], dtype=np.float32)

    sd = np.nanstd(values_2d, axis=0, ddof=1)

    if mode.lower() == "sem":
        return sd / np.sqrt(values_2d.shape[0])

    return sd

# New helper function for updated individual channel sheets with replicates side by side and mean/variability columns
#this exports both normalized (max intensity=1) and grid normalized profiles 
def export_radial_profiles_by_channel(
    all_profile_rows,
    out_dir,
    channel_names=None,
    error_mode="sd",
    profile_key="radial_mean_norm",
    workbook_label="normalized"
):
    """
    Export one Excel workbook with one sheet per channel.

    profile_key options:
      - "radial_mean_norm"   = Makis-style profile, each channel max = 1
      - "radial_mean_global" = non-Makis/spiderweb-normalized profile, preserves relative differences but not max=1 
      - "radial_std_norm"    = normalized angular variability
      - "radial_std_global"  = global angular variability

    Each sheet contains:
    r_frac | image_1 | image_2 | ... | mean | sd/sem | n
    """
    if not all_profile_rows:
        print("⚠️ No radial profiles collected for Excel export.")
        return None

    excel_dir = os.path.join(out_dir, "excel")
    os.makedirs(excel_dir, exist_ok=True)

    workbook_path = os.path.join(
        excel_dir,
        f"Patterna_radial_profiles_by_channel_{workbook_label}.xlsx"
    )

    df_all = pd.DataFrame(all_profile_rows)

    if profile_key not in df_all.columns:
        raise ValueError(
            f"'{profile_key}' was not found in all_profile_rows. "
            f"Available columns: {list(df_all.columns)}"
        )

    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        for ch in sorted(df_all["channel"].unique()):
            df_ch = df_all[df_all["channel"] == ch].copy()

            if len(df_ch) == 0:
                continue

            channel_name = df_ch["channel_name"].iloc[0]
            sheet_name = str(channel_name)[:31]

            r_fracs = np.asarray(df_ch["r_fracs"].iloc[0], dtype=np.float32)

            out = pd.DataFrame({"r_frac": r_fracs})
            replicate_profiles = []

            for _, row in df_ch.iterrows():
                image_name = str(row["image"])
                profile = np.asarray(row[profile_key], dtype=np.float32)

                if len(profile) != len(r_fracs):
                    print(f"⚠️ Skipping profile with mismatched length: {image_name}, channel {ch}")
                    continue

                safe_col = image_name[:80]
                out[safe_col] = profile
                replicate_profiles.append(profile)

            if replicate_profiles:
                arr = np.vstack(replicate_profiles)
                out["mean"] = np.nanmean(arr, axis=0)
                out[error_mode.lower()] = error_curve(arr, mode=error_mode)
                out["n"] = arr.shape[0]

            out.to_excel(writer, sheet_name=sheet_name, index=False)

        metadata = pd.DataFrame({
            "setting": [
                "profile_key",
                "workbook_label",
                "error_mode",
                "n_profiles_total",
                "n_channels",
                "note",
            ],
            "value": [
                profile_key,
                workbook_label,
                error_mode,
                len(all_profile_rows),
                df_all["channel"].nunique(),
                "Each channel sheet contains all MP radial profiles side by side plus mean and variability.",
            ],
        })
        metadata.to_excel(writer, sheet_name="metadata", index=False)

    print(f"✔️ Exported radial profile workbook: {workbook_path}")
    return workbook_path
#new helper for groiuped radial profile plots
def save_grouped_radial_profile_plots(
    all_profile_rows,
    out_dir,
    channel_names=None,
    error_mode="sd",
    profile_key="radial_mean_norm",
    profile_label="curve_max_normalized"
):
    """
    Save grouped radial profile plots across all MPs.

    Outputs:
      1. One combined plot with all channels: mean ± SD/SEM
      2. One plot per channel showing individual MPs + mean ± SD/SEM

    profile_key options:
      - radial_mean_norm: curve-max normalized, each radial curve max = 1
      - radial_mean_global: Patterna grid 0–1 normalized profile
    """
    if not all_profile_rows:
        print(f"⚠️ No profiles available for grouped radial plots: {profile_label}")
        return

    grouped_dir = os.path.join(out_dir, "grouped_profiles")
    os.makedirs(grouped_dir, exist_ok=True)

    df_all = pd.DataFrame(all_profile_rows)

    if profile_key not in df_all.columns:
        raise ValueError(
            f"'{profile_key}' not found in all_profile_rows. "
            f"Available columns: {list(df_all.columns)}"
        )

    # -----------------------------
    # 1) Combined all-channel plot
    # -----------------------------
    plt.figure(figsize=(8, 5))

    for ch in sorted(df_all["channel"].unique()):
        df_ch = df_all[df_all["channel"] == ch].copy()

        if len(df_ch) == 0:
            continue

        channel_name = df_ch["channel_name"].iloc[0]
        r_fracs = np.asarray(df_ch["r_fracs"].iloc[0], dtype=np.float32)

        profiles = []
        for _, row in df_ch.iterrows():
            profile = np.asarray(row[profile_key], dtype=np.float32)
            if len(profile) == len(r_fracs):
                profiles.append(profile)

        if not profiles:
            continue

        arr = np.vstack(profiles)
        mean_curve = np.nanmean(arr, axis=0)
        err_curve = error_curve(arr, mode=error_mode)

        color = get_channel_color(ch)

        plt.plot(
            r_fracs,
            mean_curve,
            linewidth=2.4,
            color=color,
            label=f"{channel_name} mean"
        )
        plt.fill_between(
            r_fracs,
            mean_curve - err_curve,
            mean_curve + err_curve,
            color=color,
            alpha=0.18
        )

    plt.title(f"Grouped radial profiles — {profile_label}")
    plt.xlabel("Radius fraction, centre → edge")
    plt.ylabel("Normalized intensity")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        os.path.join(grouped_dir, f"grouped_radial_profiles_all_channels_{profile_label}.svg")
    )
    plt.savefig(
        os.path.join(grouped_dir, f"grouped_radial_profiles_all_channels_{profile_label}.png"),
        dpi=300
    )
    plt.close()

    # -----------------------------
    # 2) One plot per channel
    # -----------------------------
    for ch in sorted(df_all["channel"].unique()):
        df_ch = df_all[df_all["channel"] == ch].copy()

        if len(df_ch) == 0:
            continue

        channel_name = df_ch["channel_name"].iloc[0]
        safe_channel = str(channel_name).replace("/", "_").replace(" ", "_")
        r_fracs = np.asarray(df_ch["r_fracs"].iloc[0], dtype=np.float32)

        profiles = []

        plt.figure(figsize=(7, 4.5))
        color = get_channel_color(ch)

        for _, row in df_ch.iterrows():
            image_name = str(row["image"])
            profile = np.asarray(row[profile_key], dtype=np.float32)

            if len(profile) != len(r_fracs):
                continue

            profiles.append(profile)

            # individual MPs as faint lines
            plt.plot(
                r_fracs,
                profile,
                color=color,
                alpha=0.22,
                linewidth=1.0
            )

        if not profiles:
            plt.close()
            continue

        arr = np.vstack(profiles)
        mean_curve = np.nanmean(arr, axis=0)
        err_curve = error_curve(arr, mode=error_mode)

        # mean ± error
        plt.plot(
            r_fracs,
            mean_curve,
            color=color,
            linewidth=2.8,
            label=f"{channel_name} mean"
        )

        plt.fill_between(
            r_fracs,
            mean_curve - err_curve,
            mean_curve + err_curve,
            color=color,
            alpha=0.22,
            label=f"± {error_mode.upper()}"
        )

        plt.title(f"Grouped radial profile — {channel_name} — {profile_label}")
        plt.xlabel("Radius fraction, centre → edge")
        plt.ylabel("Normalized intensity")
        plt.grid(False)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend(fontsize=8)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                grouped_dir,
                f"grouped_radial_profile_ch{ch}_{safe_channel}_{profile_label}.svg"
            )
        )
        plt.savefig(
            os.path.join(
                grouped_dir,
                f"grouped_radial_profile_ch{ch}_{safe_channel}_{profile_label}.png"
            ),
            dpi=300
        )
        plt.close()

    print(f"✔️ Saved grouped radial profile plots: {profile_label}")
#helper for grouped grids 
def load_and_average_saved_grids(out_dir):
    """
    Load all saved spiderweb grids from out_dir/grids/*_grid.npy
    and compute the average grid across MPs.

    Returns:
      mean_grid: (C, R, A)
      sd_grid:   (C, R, A)
      n_grids:   int
    """
    grid_dir = os.path.join(out_dir, "grids")
    grid_paths = sorted(glob.glob(os.path.join(grid_dir, "*_grid.npy")))

    if not grid_paths:
        print("⚠️ No saved grids found for grouped heatmaps.")
        return None, None, 0

    grids = []
    target_shape = None

    for gp in grid_paths:
        try:
            g = np.load(gp)
        except Exception as e:
            print(f"⚠️ Could not load grid: {gp} -> {e}")
            continue

        if g.ndim != 3:
            print(f"⚠️ Skipping grid with unexpected shape: {gp} shape={g.shape}")
            continue

        if target_shape is None:
            target_shape = g.shape

        if g.shape != target_shape:
            print(f"⚠️ Skipping grid with mismatched shape: {gp} shape={g.shape}, expected={target_shape}")
            continue

        grids.append(g.astype(np.float32))

    if not grids:
        print("⚠️ No compatible grids found for grouped heatmaps.")
        return None, None, 0

    arr = np.stack(grids, axis=0)  # (N, C, R, A)
    mean_grid = np.nanmean(arr, axis=0)

    if arr.shape[0] > 1:
        sd_grid = np.nanstd(arr, axis=0, ddof=1)
    else:
        sd_grid = np.zeros_like(mean_grid, dtype=np.float32)

    print(f"✔️ Loaded {arr.shape[0]} grids for grouped heatmaps.")
    return mean_grid, sd_grid, arr.shape[0]

#helper for grouped rectangular heatmaps (R vs angle)
def save_grouped_rectangular_heatmaps(mean_grid, out_dir, r_fracs, channel_names=None, label="mean"):
    """
    Save grouped rectangular heatmaps per channel.
    mean_grid: (C, R, A)
    """
    grouped_dir = os.path.join(out_dir, "grouped_heatmaps")
    os.makedirs(grouped_dir, exist_ok=True)

    C = mean_grid.shape[0]

    for ch in range(C):
        channel_name = CHANNEL_NAMES.get(ch, f"ch{ch}") if channel_names is None else channel_names.get(ch, f"ch{ch}")
        safe_channel = str(channel_name).replace("/", "_").replace(" ", "_")

        plt.figure(figsize=(8, 4.2))
        plt.imshow(
            mean_grid[ch],
            aspect="auto",
            origin="lower",
            extent=[0, 360, float(r_fracs[0]), float(r_fracs[-1])],
            vmin=0,
            vmax=1,
            cmap=get_channel_cmap(ch)
        )
        plt.colorbar(label="Normalized intensity")
        plt.title(f"Grouped rectangular heatmap — {channel_name}")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Radius fraction")
        plt.tight_layout()

        base = os.path.join(
            grouped_dir,
            f"grouped_rectangular_heatmap_ch{ch}_{safe_channel}_{label}"
        )

        plt.savefig(base + ".svg")
        plt.savefig(base + ".png", dpi=300)
        plt.close()

    print("✔️ Saved grouped rectangular heatmaps.")

#helper for grouped polar heatmaps (spiderweb grid rendered into circle)
def save_grouped_polar_heatmaps(mean_grid, out_dir, angles, r_fracs, channel_names=None, label="mean"):
    """
    Save grouped polar heatmaps per channel.
    mean_grid: (C, R, A)
    """
    grouped_dir = os.path.join(out_dir, "grouped_topography_maps")
    os.makedirs(grouped_dir, exist_ok=True)

    C = mean_grid.shape[0]

    if len(angles) > 1:
        dtheta = float(np.mean(np.diff(angles)))
    else:
        dtheta = 2 * np.pi

    theta_edges = np.concatenate([angles, [angles[-1] + dtheta]])

    if len(r_fracs) > 1:
        dr = float(np.mean(np.diff(r_fracs)))
    else:
        dr = 1.0

    r_edges = np.concatenate([
        [max(0.0, r_fracs[0] - dr / 2)],
        (r_fracs[:-1] + r_fracs[1:]) / 2,
        [r_fracs[-1] + dr / 2]
    ])

    for ch in range(C):
        channel_name = CHANNEL_NAMES.get(ch, f"ch{ch}") if channel_names is None else channel_names.get(ch, f"ch{ch}")
        safe_channel = str(channel_name).replace("/", "_").replace(" ", "_")

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="polar")

        pcm = ax.pcolormesh(
            theta_edges,
            r_edges,
            mean_grid[ch],
            shading="auto",
            vmin=0,
            vmax=1,
            cmap=get_channel_cmap(ch),
            rasterized=True
        )

        ax.set_title(f"Grouped polar heatmap — {channel_name}", pad=20)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 1.0)
        ax.grid(alpha=0.25)

        cbar = plt.colorbar(pcm, ax=ax, pad=0.12)
        cbar.set_label("Normalized intensity")

        plt.tight_layout()

        base = os.path.join(
            grouped_dir,
            f"grouped_polar_heatmap_ch{ch}_{safe_channel}_{label}"
        )

        plt.savefig(base + ".pdf", bbox_inches="tight", dpi=300)
        plt.savefig(base + ".png", bbox_inches="tight", dpi=600)
        plt.close()

    print("✔️ Saved grouped polar heatmaps.")
#helper to load binary mask from multiple formats (tif/tiff/png/jpg) for grouped orthogonal profiles
def load_mask_any(mask_path):
    """
    Load a binary mask from tif/tiff/png/jpg.
    Returns uint8 mask with values 0/1.
    """
    ext = os.path.splitext(mask_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        m = tiff.imread(mask_path)
    else:
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if m is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    m = np.asarray(m)
    if m.ndim > 2:
        m = m[..., 0]

    return (m > 0).astype(np.uint8)

#helper to find mask for given grid stem (try multiple extensions)
def find_mask_for_stem(out_dir, stem):
    """
    Try to find the saved mask corresponding to one saved grid stem.
    """
    candidates = [
        os.path.join(out_dir, "masks", f"{stem}_mask.tif"),
        os.path.join(out_dir, "masks", f"{stem}_mask.tiff"),
        os.path.join(out_dir, "masks", f"{stem}_mask.png"),
        os.path.join(out_dir, "masks", f"{stem}_mask.jpg"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return None

#helper to load grouped orthogonal profile data from saved grids + masks, returning list of dicts for plotting and analysis
def load_grouped_orthogonal_profiles(out_dir, channel_names=None, half_width_bins=3):
    """
    Build grouped orthogonal profile data from saved grids + saved masks.

    Returns:
      rows: list of dicts, one per image per channel
    """
    grid_dir = os.path.join(out_dir, "grids")
    grid_paths = sorted(glob.glob(os.path.join(grid_dir, "*_grid.npy")))

    if not grid_paths:
        print("⚠️ No saved grids found for grouped orthogonal profiles.")
        return []

    rows = []

    for gp in grid_paths:
        stem = os.path.basename(gp).replace("_grid.npy", "")
        mask_path = find_mask_for_stem(out_dir, stem)

        if mask_path is None:
            print(f"⚠️ No matching mask found for {stem}; skipping grouped orthogonal.")
            continue

        try:
            grid = np.load(gp).astype(np.float32)   # (C,R,A)
            mask = load_mask_any(mask_path)
        except Exception as e:
            print(f"⚠️ Failed loading grouped orthogonal input for {stem}: {e}")
            continue

        if grid.ndim != 3:
            print(f"⚠️ Skipping malformed grid for {stem}: shape={grid.shape}")
            continue

        C, R, A = grid.shape
        r_fracs = np.linspace(0.0, 1.0, R).astype(np.float32)

        theta0 = principal_axis_angle(mask)

        for ch in range(C):
            channel_name = CHANNEL_NAMES.get(ch, f"ch{ch}") if channel_names is None else channel_names.get(ch, f"ch{ch}")

            p_major, p_minor = orthogonal_profiles_from_grid(
                grid[ch],
                theta0,
                A,
                half_width_bins=half_width_bins
            )

            rows.append({
                "image": stem,
                "channel": int(ch),
                "channel_name": channel_name,
                "r_fracs": r_fracs.tolist(),
                "p_major": p_major.tolist(),
                "p_minor": p_minor.tolist(),
            })

    print(f"✔️ Loaded grouped orthogonal profiles from {len(rows)} channel-profile entries.")
    return rows

#helper for grouped orthogonal profile plots (major/minor profiles per channel with mean/variability)
def save_grouped_orthogonal_plots(
    orthogonal_rows,
    out_dir,
    channel_names=None,
    error_mode="sd",
    label="grouped"
):
    """
    Save grouped orthogonal plots per channel:
      - faint individual major/minor profiles
      - mean ± SD/SEM for major and minor
    """
    if not orthogonal_rows:
        print("⚠️ No grouped orthogonal rows available.")
        return

    grouped_dir = os.path.join(out_dir, "grouped_orthogonal")
    os.makedirs(grouped_dir, exist_ok=True)

    df_all = pd.DataFrame(orthogonal_rows)

    for ch in sorted(df_all["channel"].unique()):
        df_ch = df_all[df_all["channel"] == ch].copy()

        if len(df_ch) == 0:
            continue

        channel_name = df_ch["channel_name"].iloc[0]
        safe_channel = str(channel_name).replace("/", "_").replace(" ", "_")
        r_fracs = np.asarray(df_ch["r_fracs"].iloc[0], dtype=np.float32)

        major_profiles = []
        minor_profiles = []

        plt.figure(figsize=(7.5, 5))
        color = get_channel_color(ch)

        # faint individual lines
        for _, row in df_ch.iterrows():
            pmaj = np.asarray(row["p_major"], dtype=np.float32)
            pmin = np.asarray(row["p_minor"], dtype=np.float32)

            if len(pmaj) != len(r_fracs) or len(pmin) != len(r_fracs):
                continue

            major_profiles.append(pmaj)
            minor_profiles.append(pmin)

            plt.plot(r_fracs, pmaj, color=color, alpha=0.18, linewidth=1.0)
            plt.plot(r_fracs, pmin, color=color, alpha=0.18, linewidth=1.0, linestyle="--")

        if not major_profiles or not minor_profiles:
            plt.close()
            continue

        arr_maj = np.vstack(major_profiles)
        arr_min = np.vstack(minor_profiles)

        mean_maj = np.nanmean(arr_maj, axis=0)
        mean_min = np.nanmean(arr_min, axis=0)

        err_maj = error_curve(arr_maj, mode=error_mode)
        err_min = error_curve(arr_min, mode=error_mode)

        # major
        plt.plot(
            r_fracs,
            mean_maj,
            color=color,
            linewidth=2.8,
            label=f"{channel_name} major"
        )
        plt.fill_between(
            r_fracs,
            mean_maj - err_maj,
            mean_maj + err_maj,
            color=color,
            alpha=0.18
        )

        # minor
        plt.plot(
            r_fracs,
            mean_min,
            color=color,
            linewidth=2.8,
            linestyle="--",
            label=f"{channel_name} minor"
        )
        plt.fill_between(
            r_fracs,
            mean_min - err_min,
            mean_min + err_min,
            color=color,
            alpha=0.10
        )

        plt.title(f"Grouped orthogonal profiles — {channel_name}")
        plt.xlabel("Radius fraction, centre → edge")
        plt.ylabel("Normalized intensity")
        plt.grid(False)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend(fontsize=8)
        plt.tight_layout()

        base = os.path.join(
            grouped_dir,
            f"grouped_orthogonal_ch{ch}_{safe_channel}_{label}"
        )
        plt.savefig(base + ".svg")
        plt.savefig(base + ".png", dpi=300)
        plt.close()

    print("✔️ Saved grouped orthogonal plots.")
#helper to convert ring_bin integer to readable label 
def ring_label_from_bin(x):
    try:
        x = int(x)
    except Exception:
        return "unknown"

    if x == 0:
        return "inner"
    if x == 1:
        return "middle"
    if x == 2:
        return "outer"
    return "outside"

#helper to build marker-positive thresholds from cell table, with options for global or per-image percentile thresholds, returning DataFrame with channel, channel_name, stat_col, threshold, threshold_mode, percentile
def _as_nonempty_list(x):
    """Return x as a clean list of non-empty strings."""
    if x is None:
        return []
    if isinstance(x, str):
        x = [x]
    return [str(v).strip() for v in x if str(v).strip()]


def _match_control_images(df_cells, image_patterns):
    """
    Select cells whose image name contains any pattern in image_patterns.
    Matching is case-insensitive and substring-based.
    """
    patterns = _as_nonempty_list(image_patterns)
    if not patterns:
        return df_cells.iloc[0:0].copy(), []

    img_series = df_cells["image"].astype(str)
    mask = np.zeros(len(df_cells), dtype=bool)

    for pat in patterns:
        mask |= img_series.str.contains(pat, case=False, regex=False).to_numpy()

    matched = df_cells.loc[mask].copy()
    matched_images = sorted(matched["image"].astype(str).unique().tolist()) if len(matched) else []
    return matched, matched_images


def _control_threshold_from_values(vals, method="mean_plus_2sd"):
    """
    Calculate a threshold from control-cell intensity values.
    """
    vals = np.asarray(vals, dtype=np.float32)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan, {}

    method = str(method).lower().strip()
    mean_val = float(np.mean(vals))
    sd_val = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

    extra = {
        "control_mean": mean_val,
        "control_sd": sd_val,
        "control_n_cells": int(vals.size),
        "control_method": method,
    }

    if method == "mean_plus_2sd":
        thr = mean_val + 2.0 * sd_val
        extra["sd_multiplier"] = 2.0
        extra["control_percentile"] = np.nan

    elif method == "mean_plus_3sd":
        thr = mean_val + 3.0 * sd_val
        extra["sd_multiplier"] = 3.0
        extra["control_percentile"] = np.nan

    elif method == "percentile_95":
        thr = float(np.percentile(vals, 95))
        extra["sd_multiplier"] = np.nan
        extra["control_percentile"] = 95.0

    elif method == "percentile_99":
        thr = float(np.percentile(vals, 99))
        extra["sd_multiplier"] = np.nan
        extra["control_percentile"] = 99.0

    else:
        raise ValueError(
            f"Unknown CONTROL_THRESHOLD_METHOD: {method}. "
            "Use 'mean_plus_2sd', 'mean_plus_3sd', 'percentile_95', or 'percentile_99'."
        )

    return float(thr), extra


def build_marker_positive_thresholds(
    df_cells,
    marker_channels,
    channel_names=None,
    stat="mean",
    mode="percentile_global",
    percentile=75,
    percentile_by_channel=None,
    manual_thresholds=None,
    control_images=None,
    control_images_by_channel=None,
    control_method="mean_plus_2sd"
):
    """
    Build marker-positive thresholds from the cell table.

    Supported modes:
      - manual
      - control
      - percentile_global
      - percentile_per_image

    Returns a DataFrame with one threshold row per marker/channel
    or per marker/channel/image when using percentile_per_image.
    """
    rows = []

    if df_cells is None or len(df_cells) == 0:
        return pd.DataFrame()

    if "image" not in df_cells.columns:
        raise ValueError("df_cells must contain an 'image' column for phenotype thresholding.")

    manual_thresholds = manual_thresholds or {}
    percentile_by_channel = percentile_by_channel or {}
    control_images = _as_nonempty_list(control_images)
    control_images_by_channel = control_images_by_channel or {}

    mode = str(mode).strip().lower()

    for ch in marker_channels:
        channel_name = CHANNEL_NAMES.get(ch, f"ch{ch}") if channel_names is None else channel_names.get(ch, f"ch{ch}")
        stat_col = f"ch{ch}_{stat}"
        ch_percentile = percentile_by_channel.get(ch, percentile_by_channel.get(str(ch), None))
        if ch_percentile is None:
            ch_percentile = percentile

        if stat_col not in df_cells.columns:
            print(f"⚠️ Missing intensity column for phenotype classification: {stat_col}")
            continue

        if mode == "manual":
            thr = manual_thresholds.get(ch, manual_thresholds.get(str(ch), None))

            if thr is None:
                raise ValueError(
                    f"Manual threshold mode selected, but no threshold was provided for channel {ch} ({channel_name}). "
                    f"Set PHENOTYPE_MANUAL_THRESHOLDS[{ch}] to a number."
                )

            rows.append({
                "image": "ALL",
                "channel": int(ch),
                "channel_name": channel_name,
                "stat_col": stat_col,
                "threshold": float(thr),
                "threshold_mode": mode,
                "percentile": np.nan,
                "source_images": "manual",
                "control_method": "manual",
                "control_n_cells": np.nan,
                "control_mean": np.nan,
                "control_sd": np.nan,
                "sd_multiplier": np.nan,
                "control_percentile": np.nan,
            })

        elif mode == "control":
            channel_patterns = _as_nonempty_list(
                control_images_by_channel.get(ch, control_images_by_channel.get(str(ch), []))
            )
            patterns_to_use = channel_patterns if channel_patterns else control_images

            df_control, matched_images = _match_control_images(df_cells, patterns_to_use)

            if len(df_control) == 0:
                raise ValueError(
                    f"Control threshold mode selected, but no control cells matched channel {ch} ({channel_name}). "
                    f"Patterns used: {patterns_to_use}. "
                    "Check CONTROL_IMAGES_FOR_THRESHOLDS or CONTROL_IMAGES_FOR_THRESHOLDS_BY_CHANNEL."
                )

            vals = (
                df_control[stat_col]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=np.float32)
            )

            thr, extra = _control_threshold_from_values(vals, method=control_method)

            rows.append({
                "image": "ALL",
                "channel": int(ch),
                "channel_name": channel_name,
                "stat_col": stat_col,
                "threshold": thr,
                "threshold_mode": mode,
                "percentile": np.nan,
                "source_images": "; ".join(matched_images),
                **extra,
            })

        elif mode == "percentile_global":
            vals = df_cells[stat_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32)

            if vals.size == 0:
                continue

            thr = float(np.percentile(vals, ch_percentile))

            rows.append({
                "image": "ALL",
                "channel": int(ch),
                "channel_name": channel_name,
                "stat_col": stat_col,
                "threshold": thr,
                "threshold_mode": mode,
                "percentile": float(ch_percentile),
                "source_images": "ALL",
                "control_method": "percentile_global",
                "control_n_cells": int(vals.size),
                "control_mean": float(np.mean(vals)),
                "control_sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "sd_multiplier": np.nan,
                "control_percentile": float(ch_percentile),
            })

        elif mode == "percentile_per_image":
            for image, df_img in df_cells.groupby("image"):
                vals = df_img[stat_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32)

                if vals.size == 0:
                    continue

                thr = float(np.percentile(vals, ch_percentile))

                rows.append({
                    "image": image,
                    "channel": int(ch),
                    "channel_name": channel_name,
                    "stat_col": stat_col,
                    "threshold": thr,
                    "threshold_mode": mode,
                    "percentile": float(ch_percentile),
                    "source_images": str(image),
                    "control_method": "percentile_per_image",
                    "control_n_cells": int(vals.size),
                    "control_mean": float(np.mean(vals)),
                    "control_sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "sd_multiplier": np.nan,
                    "control_percentile": float(ch_percentile),
                })

        else:
            raise ValueError(
                f"Unknown PHENOTYPE_THRESHOLD_MODE: {mode}. "
                "Use 'manual', 'control', 'percentile_global', or 'percentile_per_image'."
            )

    return pd.DataFrame(rows)


#helper to classify cell phenotypes based on marker positivity, adding boolean columns and a phenotype_class column, with options for global or per-image thresholds
def classify_cell_phenotypes(
    df_cells,
    marker_channels,
    channel_names=None,
    stat="mean",
    threshold_mode="percentile_global",
    percentile=75,
    percentile_by_channel=None,
    manual_thresholds=None,
    control_images=None,
    control_images_by_channel=None,
    control_method="mean_plus_2sd"
):
    """
    Add marker-positive boolean columns and a phenotype_class column.

    phenotype_class examples:
      negative
      TBXT only
      SOX2 only
      GATA6 only
      TBXT+SOX2
      TBXT+SOX2+GATA6
    """
    if df_cells is None or len(df_cells) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df = df_cells.copy()
    threshold_mode = str(threshold_mode).strip().lower()

    # Add readable ring labels
    if "ring_bin" in df.columns:
        df["ring_label"] = df["ring_bin"].apply(ring_label_from_bin)
    else:
        df["ring_label"] = "unknown"

    thresholds = build_marker_positive_thresholds(
        df,
        marker_channels=marker_channels,
        channel_names=channel_names,
        stat=stat,
        mode=threshold_mode,
        percentile=percentile,
        percentile_by_channel=percentile_by_channel,
        manual_thresholds=manual_thresholds,
        control_images=control_images,
        control_images_by_channel=control_images_by_channel,
        control_method=control_method
    )

    if thresholds is None or len(thresholds) == 0:
        print("⚠️ No thresholds generated for cell phenotype classification.")
        return df, thresholds

    # Apply thresholds
    for ch in marker_channels:
        channel_name = CHANNEL_NAMES.get(ch, f"ch{ch}") if channel_names is None else channel_names.get(ch, f"ch{ch}")
        stat_col = f"ch{ch}_{stat}"
        pos_col = f"{channel_name}_positive"

        if stat_col not in df.columns:
            continue

        if threshold_mode in ["manual", "control", "percentile_global"]:
            match = thresholds[(thresholds["channel"] == ch) & (thresholds["image"] == "ALL")]
            if len(match) == 0:
                continue

            thr = float(match["threshold"].iloc[0])
            df[pos_col] = df[stat_col] >= thr
            df[f"{channel_name}_threshold_used"] = thr

        elif threshold_mode == "percentile_per_image":
            df[pos_col] = False
            df[f"{channel_name}_threshold_used"] = np.nan

            for image, df_img in df.groupby("image"):
                match = thresholds[
                    (thresholds["channel"] == ch) &
                    (thresholds["image"] == image)
                ]

                if len(match) == 0:
                    continue

                thr = float(match["threshold"].iloc[0])
                idx = df["image"] == image
                df.loc[idx, pos_col] = df.loc[idx, stat_col] >= thr
                df.loc[idx, f"{channel_name}_threshold_used"] = thr

    marker_names = [
        CHANNEL_NAMES.get(ch, f"ch{ch}") if channel_names is None else channel_names.get(ch, f"ch{ch}")
        for ch in marker_channels
    ]
    positive_cols = [f"{name}_positive" for name in marker_names if f"{name}_positive" in df.columns]

    def _phenotype_label(row):
        positive = []
        for name in marker_names:
            col = f"{name}_positive"
            if col in row.index and bool(row[col]):
                positive.append(name)

        if len(positive) == 0:
            return "negative"
        if len(positive) == 1:
            return f"{positive[0]} only"
        return "+".join(positive)

    df["n_positive_markers"] = df[positive_cols].sum(axis=1).astype(int) if positive_cols else 0
    df["phenotype_class"] = df.apply(_phenotype_label, axis=1)

    return df, thresholds


#helper to summarize classified cell phenotypes into tables by image, ring, and channel positivity
def summarize_cell_phenotypes(df_classified, marker_channels, channel_names=None):
    """
    Build useful summary tables:
      - summary_by_image
      - summary_by_ring
      - summary_by_channel
    """
    if df_classified is None or len(df_classified) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_classified.copy()

    # Summary by image and phenotype class
    summary_by_image = (
        df.groupby(["image", "phenotype_class"])
        .size()
        .reset_index(name="n_cells")
    )
    totals_img = df.groupby("image").size().reset_index(name="total_cells")
    summary_by_image = summary_by_image.merge(totals_img, on="image", how="left")
    summary_by_image["fraction_cells"] = summary_by_image["n_cells"] / summary_by_image["total_cells"]

    # Summary by ring and phenotype class
    summary_by_ring = (
        df.groupby(["ring_label", "phenotype_class"])
        .size()
        .reset_index(name="n_cells")
    )
    totals_ring = df.groupby("ring_label").size().reset_index(name="total_cells")
    summary_by_ring = summary_by_ring.merge(totals_ring, on="ring_label", how="left")
    summary_by_ring["fraction_cells"] = summary_by_ring["n_cells"] / summary_by_ring["total_cells"]

    # Summary by marker/channel and ring
    rows = []
    for ch in marker_channels:
        channel_name = CHANNEL_NAMES.get(ch, f"ch{ch}") if channel_names is None else channel_names.get(ch, f"ch{ch}")
        pos_col = f"{channel_name}_positive"

        if pos_col not in df.columns:
            continue

        for ring, df_ring in df.groupby("ring_label"):
            n_total = len(df_ring)
            n_pos = int(df_ring[pos_col].sum())
            rows.append({
                "channel": int(ch),
                "channel_name": channel_name,
                "ring_label": ring,
                "n_positive": n_pos,
                "n_total": n_total,
                "fraction_positive": n_pos / n_total if n_total > 0 else np.nan,
            })

    summary_by_channel = pd.DataFrame(rows)

    return summary_by_image, summary_by_ring, summary_by_channel

#helper to export cell phenotype classification and summaries into one Excel workbook
def export_cell_phenotype_workbook(
    df_classified,
    thresholds,
    summary_by_image,
    summary_by_ring,
    summary_by_channel,
    out_dir
):
    """
    Export cell phenotype results into one Excel workbook.
    """
    phenotype_dir = os.path.join(out_dir, "cell_phenotypes")
    os.makedirs(phenotype_dir, exist_ok=True)

    workbook_path = os.path.join(
        phenotype_dir,
        "Patterna_cell_phenotype_classification.xlsx"
    )

    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        df_classified.to_excel(writer, sheet_name="classified_cells", index=False)
        summary_by_image.to_excel(writer, sheet_name="summary_by_image", index=False)
        summary_by_ring.to_excel(writer, sheet_name="summary_by_ring", index=False)
        summary_by_channel.to_excel(writer, sheet_name="summary_by_channel", index=False)
        thresholds.to_excel(writer, sheet_name="thresholds", index=False)

    print(f"✔️ Exported cell phenotype workbook: {workbook_path}")
    return workbook_path

#helper to save phenotype summary plots (stacked bars by ring/image, positive fraction by marker/ring)

def save_marker_intensity_threshold_diagnostics(
    df_classified,
    thresholds,
    out_dir,
    marker_channels=None,
    channel_names=None,
    stat="mean"
):
    """
    Save diagnostic plots showing all per-cell marker intensities and the positivity cutoff.
    This helps visually inspect whether each marker threshold is too strict/permissive.
    """
    if df_classified is None or len(df_classified) == 0:
        print("⚠️ No classified cells available for threshold diagnostic plots.")
        return

    if marker_channels is None:
        marker_channels = PHENOTYPE_MARKER_CHANNELS

    if channel_names is None:
        channel_names = CHANNEL_NAMES

    phenotype_dir = os.path.join(out_dir, "cell_phenotypes")
    os.makedirs(phenotype_dir, exist_ok=True)

    n = len(marker_channels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.2 * n), squeeze=False)
    axes = axes.ravel()

    rows = []

    for ax, ch in zip(axes, marker_channels):
        marker_name = channel_names.get(ch, f"ch{ch}")
        stat_col = f"ch{ch}_{stat}"
        pos_col = f"{marker_name}_positive"

        if stat_col not in df_classified.columns:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Missing {stat_col}", ha="center", va="center")
            continue

        vals = (
            df_classified[stat_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=np.float32)
        )

        if vals.size == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No values for {marker_name}", ha="center", va="center")
            continue

        # Find global/manual/control threshold if available
        thr = np.nan
        if thresholds is not None and len(thresholds) > 0:
            match = thresholds[(thresholds["channel"] == ch) & (thresholds["image"] == "ALL")]
            if len(match) > 0:
                thr = float(match["threshold"].iloc[0])

        color = get_channel_color(ch)

        # Histogram of all cell intensities
        ax.hist(vals, bins=60, color=color, alpha=0.55, edgecolor="none")
        if np.isfinite(thr):
            ax.axvline(thr, color="black", linewidth=2.2, linestyle="--", label=f"cutoff = {thr:.2f}")

        n_total = int(len(df_classified))
        n_pos = int(df_classified[pos_col].sum()) if pos_col in df_classified.columns else 0
        frac_pos = n_pos / n_total if n_total > 0 else np.nan

        ax.set_title(f"{marker_name} cell-intensity threshold diagnostic")
        ax.set_xlabel(f"{marker_name} intensity ({stat})")
        ax.set_ylabel("Number of cells")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

        ax.text(
            0.98, 0.95,
            f"positive: {n_pos}/{n_total} ({frac_pos:.2%})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="#CCCCCC")
        )

        rows.append({
            "channel": int(ch),
            "channel_name": marker_name,
            "stat_col": stat_col,
            "threshold": thr,
            "n_total": n_total,
            "n_positive": n_pos,
            "fraction_positive": frac_pos,
            "intensity_min": float(np.min(vals)),
            "intensity_p25": float(np.percentile(vals, 25)),
            "intensity_median": float(np.median(vals)),
            "intensity_p75": float(np.percentile(vals, 75)),
            "intensity_p90": float(np.percentile(vals, 90)),
            "intensity_p95": float(np.percentile(vals, 95)),
            "intensity_p99": float(np.percentile(vals, 99)),
            "intensity_max": float(np.max(vals)),
        })

    plt.tight_layout()
    base = os.path.join(phenotype_dir, "marker_intensity_threshold_diagnostics")
    plt.savefig(base + ".svg")
    plt.savefig(base + ".png", dpi=300)
    plt.close(fig)

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(phenotype_dir, "marker_intensity_threshold_diagnostics.csv"),
            index=False
        )

    print("✔️ Saved marker intensity threshold diagnostics.")


def save_cell_phenotype_plots(
    summary_by_image,
    summary_by_ring,
    summary_by_channel,
    out_dir,
    channel_names=None
):
    """
    Save phenotype classification plots using the Patterna palette.
    """
    phenotype_dir = os.path.join(out_dir, "cell_phenotypes")
    os.makedirs(phenotype_dir, exist_ok=True)

    # -----------------------------
    # Plot 1: stacked phenotype fractions by ring
    # -----------------------------
    if summary_by_ring is not None and len(summary_by_ring) > 0:
        pivot = summary_by_ring.pivot_table(
            index="ring_label",
            columns="phenotype_class",
            values="fraction_cells",
            fill_value=0
        )

        ring_order = [r for r in ["inner", "middle", "outer", "outside", "unknown"] if r in pivot.index]
        pivot = pivot.loc[ring_order]

        phenotype_colors = [
            get_phenotype_color(col, channel_names=channel_names)
            for col in pivot.columns
        ]

        ax = pivot.plot(
            kind="bar",
            stacked=True,
            figsize=(8, 5),
            width=0.8,
            color=phenotype_colors
        )

        ax.set_title("Cell phenotype distribution by radial zone")
        ax.set_xlabel("Radial zone")
        ax.set_ylabel("Fraction of cells")
        ax.set_ylim(0, 1.0)
        ax.legend(title="Phenotype class", fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        base = os.path.join(phenotype_dir, "cell_phenotype_stacked_by_ring")
        plt.savefig(base + ".svg")
        plt.savefig(base + ".png", dpi=300)
        plt.close()

    # -----------------------------
    # Plot 2: stacked phenotype fractions by image
    # -----------------------------
    if summary_by_image is not None and len(summary_by_image) > 0:
        pivot = summary_by_image.pivot_table(
            index="image",
            columns="phenotype_class",
            values="fraction_cells",
            fill_value=0
        )

        phenotype_colors = [
            get_phenotype_color(col, channel_names=channel_names)
            for col in pivot.columns
        ]

        ax = pivot.plot(
            kind="bar",
            stacked=True,
            figsize=(10, 5),
            width=0.8,
            color=phenotype_colors
        )

        ax.set_title("Cell phenotype distribution by micropattern")
        ax.set_xlabel("Micropattern")
        ax.set_ylabel("Fraction of cells")
        ax.set_ylim(0, 1.0)
        ax.legend(title="Phenotype class", fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        base = os.path.join(phenotype_dir, "cell_phenotype_stacked_by_image")
        plt.savefig(base + ".svg")
        plt.savefig(base + ".png", dpi=300)
        plt.close()

    # -----------------------------
    # Plot 3: fraction positive by marker and ring
    # -----------------------------
    if summary_by_channel is not None and len(summary_by_channel) > 0:
        pivot = summary_by_channel.pivot_table(
            index="ring_label",
            columns="channel_name",
            values="fraction_positive",
            fill_value=0
        )

        ring_order = [r for r in ["inner", "middle", "outer", "outside", "unknown"] if r in pivot.index]
        pivot = pivot.loc[ring_order]

        # Direct marker/channel colors:
        # TBXT = clean ch1, SOX2 = clean ch2, GATA6 = clean ch3
        marker_color_map = {}
        for _, row in summary_by_channel.drop_duplicates("channel_name").iterrows():
            marker_color_map[str(row["channel_name"])] = get_channel_color(int(row["channel"]))

        marker_colors = [
            marker_color_map.get(str(col), get_marker_color(str(col), channel_names=channel_names))
            for col in pivot.columns
        ]

        ax = pivot.plot(
            kind="bar",
            figsize=(8, 5),
            width=0.8,
            color=marker_colors
        )

        ax.set_title("Marker-positive cell fraction by radial zone")
        ax.set_xlabel("Radial zone")
        ax.set_ylabel("Fraction positive")
        ax.set_ylim(0, 1.0)
        ax.legend(title="Marker", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        base = os.path.join(phenotype_dir, "positive_fraction_by_marker_ring")
        plt.savefig(base + ".svg")
        plt.savefig(base + ".png", dpi=300)
        plt.close()

    print("✔️ Saved cell phenotype plots.")
#helper to save integrated spatial summary panels per image, combining phenotype map, topography map, radial profiles, phenotype distribution, marker positivity, and metadata
def save_integrated_spatial_summary_panels(
    df_classified,
    out_dir,
    channel_names=None,
    marker_channels=None
):
    """
    Save per-image integrated spatial summary panels.

    Layout:
      Row 1: TBXT / SOX2 / GATA6 cell-intensity maps + phenotype map
      Row 2: marker-positive fraction by radial zone / phenotype distribution by radial zone /
             phenotype-class distribution across the MP / radial intensity profiles
      Row 3: marker-positive fraction across the whole MP / phenotype legend /
             metadata summary / major-minor axis overlay + AI values
    """
    if df_classified is None or len(df_classified) == 0:
        print("⚠️ No classified cells available for integrated summary panels.")
        return

    if marker_channels is None:
        marker_channels = PHENOTYPE_MARKER_CHANNELS

    if channel_names is None:
        channel_names = CHANNEL_NAMES

    panel_dir = os.path.join(out_dir, "summary_panels")
    os.makedirs(panel_dir, exist_ok=True)

    grid_dir = os.path.join(out_dir, "grids")
    grid_paths = sorted(glob.glob(os.path.join(grid_dir, "*_grid.npy")))

    if not grid_paths:
        print("⚠️ No saved grids found for integrated summary panels.")
        return

    ring_order = ["inner", "middle", "outer", "outside", "unknown"]
    pheno_order = [
        "negative",
        "TBXT only",
        "SOX2 only",
        "GATA6 only",
        "TBXT+SOX2",
        "TBXT+GATA6",
        "SOX2+GATA6",
        "TBXT+SOX2+GATA6",
    ]

    # optional display limits for globally comparable cell-intensity maps
    display_limits_by_channel = None
    try:
        display_limits_by_channel = build_cell_intensity_display_limits(
            df_classified,
            marker_channels=marker_channels,
            channel_names=channel_names,
            stat=CELL_INTENSITY_MAP_STAT,
            mode=CELL_INTENSITY_NORM_MODE,
            pcts=CELL_INTENSITY_NORM_PCTS,
            manual_limits=CELL_INTENSITY_MANUAL_LIMITS,
            out_dir=out_dir
        )
    except Exception as e:
        print(f"Could not build global display limits for summary panels: {e}")
        display_limits_by_channel = {}

    def _draw_major_minor_overlay(ax, mask, grid, stem_local):
        """
        Draw major/minor axes in the real mask coordinate system.
        Do NOT use the spiderweb grid as the image background here because grid is (C,R,A),
        not the original XY image. Using the grid makes the panel appear as a tiny rectangle.
        """
        if mask is None or mask.sum() == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "No mask available", ha="center", va="center")
            return {}

        mask_u8 = (mask > 0).astype(np.uint8)
        H, W = mask_u8.shape

        # Clean neutral background: white outside, light gray inside colony.
        rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
        rgb[mask_u8 > 0] = np.array([235, 235, 235], dtype=np.uint8)

        # Boundary
        try:
            bnd = find_boundaries(mask_u8, mode="thick")
            rgb[bnd] = np.array([40, 40, 40], dtype=np.uint8)
        except Exception:
            pass

        ax.imshow(rgb)

        ys, xs = np.nonzero(mask_u8 > 0)
        if len(xs) < 10:
            ax.set_title("Major/minor axes")
            ax.axis("off")
            return {}

        cx = float(xs.mean())
        cy = float(ys.mean())
        X = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        X -= X.mean(axis=0, keepdims=True)
        cov = (X.T @ X) / (X.shape[0] + 1e-8)
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w = w[order]
        v = v[:, order]

        v_major = v[:, 0]
        v_minor = v[:, 1]

        # Lengths scaled to the colony bounding box so the axes are visible.
        x_span = float(xs.max() - xs.min())
        y_span = float(ys.max() - ys.min())
        axis_scale = 0.48 * max(x_span, y_span)

        major_color = "#2563EB"  # blue
        minor_color = "#F97316"  # orange

        x0m, y0m = cx - v_major[0] * axis_scale, cy - v_major[1] * axis_scale
        x1m, y1m = cx + v_major[0] * axis_scale, cy + v_major[1] * axis_scale
        x0n, y0n = cx - v_minor[0] * axis_scale, cy - v_minor[1] * axis_scale
        x1n, y1n = cx + v_minor[0] * axis_scale, cy + v_minor[1] * axis_scale

        ax.plot([x0m, x1m], [y0m, y1m], color=major_color, linewidth=3.0, label="major axis")
        ax.plot([x0n, x1n], [y0n, y1n], color=minor_color, linewidth=3.0, label="minor axis")
        ax.scatter([cx], [cy], s=18, color="white", edgecolors="black", linewidths=0.6, zorder=5)

        ax.set_title("Major/minor axes + AI")
        ax.set_xlim(xs.min() - 20, xs.max() + 20)
        ax.set_ylim(ys.max() + 20, ys.min() - 20)
        ax.set_aspect("equal")
        ax.axis("off")

        ai_rows = {}
        if grid is not None and grid.ndim == 3:
            Cg, Rg, Ag = grid.shape
            theta0 = principal_axis_angle(mask_u8)
            for ch in marker_channels:
                if ch >= Cg:
                    continue
                pmaj, pmin = orthogonal_profiles_from_grid(grid[ch], theta0, Ag, half_width_bins=3)
                ai_rows[channel_names.get(ch, f"ch{ch}")] = float(anisotropy_index(pmaj, pmin))

        return ai_rows


    for gp in grid_paths:
        stem = os.path.basename(gp).replace("_grid.npy", "")
        df_img = df_classified[df_classified["image"] == stem].copy()

        if len(df_img) == 0:
            print(f"⚠️ No classified cells found for {stem}; skipping summary panel.")
            continue

        try:
            grid = np.load(gp).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Could not load grid for {stem}: {e}")
            continue

        C, R, A = grid.shape
        r_fracs = np.linspace(0.0, 1.0, R).astype(np.float32)
        radial_mean = grid.mean(axis=2)

        label_path = find_label_path(out_dir, stem)
        labels = None
        if label_path is not None:
            try:
                labels = load_instance_labels(label_path)
            except Exception as e:
                print(f"⚠️ Could not load labels for {stem}: {e}")
                labels = None

        mask_path = find_mask_for_stem(out_dir, stem)
        mask = None
        if mask_path is not None:
            try:
                mask = load_mask_any(mask_path)
            except Exception as e:
                print(f"⚠️ Could not load mask for {stem}: {e}")
                mask = None

        fig = plt.figure(figsize=(18, 13))
        gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 0.9], wspace=0.35, hspace=0.45)

        # ---------------- Row 1: 3 intensity maps + phenotype map ----------------
        row1_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
        for i, ch in enumerate(marker_channels[:3]):
            ax = row1_axes[i]
            marker_name = channel_names.get(ch, f"ch{ch}")
            if labels is None:
                ax.text(0.5, 0.5, 'No labels available', ha='center', va='center')
                ax.axis('off')
                continue
            try:
                rgb, cmap, lo, hi = render_single_marker_cell_map_rgb(
                    labels=labels,
                    df_img=df_img,
                    ch=ch,
                    stat=CELL_INTENSITY_MAP_STAT,
                    bg=CELL_INTENSITY_MAP_BG,
                    norm_mode=CELL_INTENSITY_NORM_MODE,
                    pcts=CELL_INTENSITY_NORM_PCTS,
                    display_limits=(display_limits_by_channel or {}).get(int(ch))
                )
                if mask is not None:
                    rgb = mask_boundary_overlay(rgb, mask, color=(0, 0, 0) if str(CELL_INTENSITY_MAP_BG).lower() != 'black' else (255,255,255))
                ax.imshow(rgb)
                ax.set_title(f"{marker_name} cell intensity")
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error rendering {marker_name} map\n{e}', ha='center', va='center', fontsize=8)
                ax.axis('off')

        ax_pheno_map = row1_axes[3]
        if labels is not None:
            try:
                rgb = render_phenotype_label_rgb(labels, df_img, channel_names=channel_names, bg='white')
                if mask is not None:
                    rgb = mask_boundary_overlay(rgb, mask, color=(0,0,0))
                ax_pheno_map.imshow(rgb)
                ax_pheno_map.set_title('Cell phenotype map')
                ax_pheno_map.axis('off')
            except Exception as e:
                ax_pheno_map.text(0.5, 0.5, f'Error rendering phenotype map\n{e}', ha='center', va='center', fontsize=8)
                ax_pheno_map.axis('off')
        else:
            ax_pheno_map.text(0.5, 0.5, 'No labels available', ha='center', va='center')
            ax_pheno_map.axis('off')

        # ---------------- Row 2 ----------------
        ax_marker_ring = fig.add_subplot(gs[1, 0])
        ax_pheno_ring = fig.add_subplot(gs[1, 1])
        ax_pheno_class = fig.add_subplot(gs[1, 2])
        ax_profile = fig.add_subplot(gs[1, 3])

        # marker-positive fraction by radial zone
        marker_rows = []
        for ch in marker_channels:
            marker_name = channel_names.get(ch, f"ch{ch}")
            pos_col = f"{marker_name}_positive"
            if pos_col not in df_img.columns or 'ring_label' not in df_img.columns:
                continue
            for ring, df_ring in df_img.groupby('ring_label'):
                n_total = len(df_ring)
                n_pos = int(df_ring[pos_col].sum())
                marker_rows.append({
                    'marker': marker_name,
                    'channel': int(ch),
                    'ring_label': ring,
                    'fraction_positive': n_pos / n_total if n_total > 0 else np.nan,
                })
        if marker_rows:
            df_marker = pd.DataFrame(marker_rows)
            pivot_marker = df_marker.pivot_table(index='ring_label', columns='marker', values='fraction_positive', fill_value=0)
            valid_rings = [r for r in ring_order if r in pivot_marker.index]
            pivot_marker = pivot_marker.loc[valid_rings]
            marker_colors = []
            for marker in pivot_marker.columns:
                ch_match = next((ch for ch, nm in channel_names.items() if nm == marker), None)
                marker_colors.append(get_channel_color(ch_match) if ch_match is not None else '#444444')
            pivot_marker.plot(kind='bar', ax=ax_marker_ring, width=0.8, color=marker_colors)
            ax_marker_ring.set_ylim(0, 1)
            ax_marker_ring.set_title('Marker-positive fraction by radial zone')
            ax_marker_ring.set_xlabel('Radial zone')
            ax_marker_ring.set_ylabel('Fraction positive')
            ax_marker_ring.tick_params(axis='x', rotation=0)
            ax_marker_ring.spines['top'].set_visible(False)
            ax_marker_ring.spines['right'].set_visible(False)
            ax_marker_ring.legend(fontsize=7, title='Marker')

        # phenotype distribution by radial zone
        if 'ring_label' in df_img.columns and 'phenotype_class' in df_img.columns:
            pheno_summary = df_img.groupby(['ring_label', 'phenotype_class']).size().reset_index(name='n_cells')
            totals = df_img.groupby('ring_label').size().reset_index(name='total_cells')
            pheno_summary = pheno_summary.merge(totals, on='ring_label', how='left')
            pheno_summary['fraction_cells'] = pheno_summary['n_cells'] / pheno_summary['total_cells']
            pivot_pheno = pheno_summary.pivot_table(index='ring_label', columns='phenotype_class', values='fraction_cells', fill_value=0)
            valid_rings = [r for r in ring_order if r in pivot_pheno.index]
            pivot_pheno = pivot_pheno.loc[valid_rings]
            phenotype_colors = [get_phenotype_color(col, channel_names=channel_names) for col in pivot_pheno.columns]
            pivot_pheno.plot(kind='bar', stacked=True, ax=ax_pheno_ring, width=0.8, color=phenotype_colors, legend=False)
            ax_pheno_ring.set_ylim(0, 1)
            ax_pheno_ring.set_title('Phenotype distribution by radial zone')
            ax_pheno_ring.set_xlabel('Radial zone')
            ax_pheno_ring.set_ylabel('Fraction of cells')
            ax_pheno_ring.tick_params(axis='x', rotation=0)
            ax_pheno_ring.spines['top'].set_visible(False)
            ax_pheno_ring.spines['right'].set_visible(False)

        # phenotype class distribution across the whole MP (8 bars)
        total_cells = float(len(df_img)) if len(df_img) > 0 else 1.0
        vals = []
        colors = []
        labels_x = []
        for ph in pheno_order:
            frac = float((df_img['phenotype_class'] == ph).sum()) / total_cells if total_cells > 0 else 0.0
            vals.append(frac)
            colors.append(get_phenotype_color(ph, channel_names=channel_names))
            labels_x.append(ph.replace(' only',''))
        ax_pheno_class.bar(range(len(vals)), vals, color=colors)
        ax_pheno_class.set_xticks(range(len(vals)))
        ax_pheno_class.set_xticklabels(labels_x, rotation=40, ha='right', fontsize=7)
        ax_pheno_class.set_ylim(0, 1)
        ax_pheno_class.set_title('Phenotype class distribution')
        ax_pheno_class.set_ylabel('Fraction of cells')
        ax_pheno_class.spines['top'].set_visible(False)
        ax_pheno_class.spines['right'].set_visible(False)

        # radial intensity profiles
        for ch in range(C):
            if ch == 0:
                continue
            label = channel_names.get(ch, f"ch{ch}")
            color = get_channel_color(ch)
            ax_profile.plot(r_fracs, radial_mean[ch], color=color, linewidth=2.2, label=label)
        ax_profile.set_title('Radial intensity profiles')
        ax_profile.set_xlabel('Radius fraction')
        ax_profile.set_ylabel('Normalized intensity')
        ax_profile.spines['top'].set_visible(False)
        ax_profile.spines['right'].set_visible(False)
        ax_profile.legend(fontsize=8)

        # ---------------- Row 3 ----------------
        ax_marker_whole = fig.add_subplot(gs[2, 0])
        ax_legend = fig.add_subplot(gs[2, 1])
        ax_text = fig.add_subplot(gs[2, 2])
        ax_axes = fig.add_subplot(gs[2, 3])

        # marker-positive fraction across the whole MP
        m_names, m_vals, m_cols = [], [], []
        for ch in marker_channels:
            marker_name = channel_names.get(ch, f"ch{ch}")
            pos_col = f"{marker_name}_positive"
            if pos_col not in df_img.columns:
                continue
            m_names.append(marker_name)
            m_vals.append(float(df_img[pos_col].sum()) / total_cells if total_cells > 0 else 0.0)
            m_cols.append(get_channel_color(ch))
        if m_names:
            ax_marker_whole.bar(m_names, m_vals, color=m_cols)
            ax_marker_whole.set_ylim(0, 1)
            ax_marker_whole.set_title('Marker-positive cells across whole MP')
            ax_marker_whole.set_ylabel('Fraction positive')
            ax_marker_whole.spines['top'].set_visible(False)
            ax_marker_whole.spines['right'].set_visible(False)

        # shared phenotype legend
        ax_legend.axis('off')
        present_classes = [ph for ph in pheno_order if ph in set(df_img['phenotype_class'].astype(str))]
        handles = [Patch(facecolor=get_phenotype_color(ph, channel_names=channel_names), edgecolor='none', label=ph) for ph in present_classes]
        if handles:
            ax_legend.legend(handles=handles, loc='upper left', fontsize=8, frameon=False, title='Phenotype class', title_fontsize=9)

        # metadata summary
        ax_text.axis('off')
        n_cells = len(df_img)
        n_negative = int((df_img['phenotype_class'] == 'negative').sum()) if 'phenotype_class' in df_img.columns else 0
        n_single = int((df_img['n_positive_markers'] == 1).sum()) if 'n_positive_markers' in df_img.columns else 0
        n_double = int((df_img['n_positive_markers'] == 2).sum()) if 'n_positive_markers' in df_img.columns else 0
        n_triple = int((df_img['n_positive_markers'] >= 3).sum()) if 'n_positive_markers' in df_img.columns else 0
        marker_lines = []
        for ch in marker_channels:
            marker_name = channel_names.get(ch, f"ch{ch}")
            pos_col = f"{marker_name}_positive"
            if pos_col in df_img.columns:
                n_pos = int(df_img[pos_col].sum())
                marker_lines.append(f"{marker_name}: {n_pos}/{n_cells} positive")
        text = (
            f"Integrated cell summary\n\n"
            f"Image: {stem}\n"
            f"N segmented cells: {n_cells}\n\n"
            f"Negative: {n_negative}\n"
            f"Single-positive: {n_single}\n"
            f"Double-positive: {n_double}\n"
            f"Triple-positive: {n_triple}\n\n"
            + "\n".join(marker_lines) + "\n\n"
            + f"Threshold mode: {PHENOTYPE_THRESHOLD_MODE}\n"
            + f"Positive percentile: {PHENOTYPE_POSITIVE_PERCENTILE}\n"
            + f"Intensity stat: {PHENOTYPE_INTENSITY_STAT}\n"
        )
        ax_text.text(0.02, 0.98, text, va='top', ha='left', fontsize=9, family='monospace')

        # major/minor axis overlay + AI
        ai_rows = _draw_major_minor_overlay(ax_axes, mask, grid, stem)
        if ai_rows:
            txt = "\n".join([f"{k}: AI={v:.3f}" for k, v in ai_rows.items()])
            ax_axes.text(0.02, 0.02, txt, transform=ax_axes.transAxes, fontsize=8,
                         va='bottom', ha='left', color='white',
                         bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.55, edgecolor='none'))
            ax_axes.legend(fontsize=7, loc='upper right', frameon=False)

        fig.suptitle(f"Patterna integrated spatial summary — {stem}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        safe_name = str(stem).replace('/', '_').replace(' ', '_')
        base = os.path.join(panel_dir, f"{safe_name}_integrated_spatial_summary")
        plt.savefig(base + '.svg')
        plt.savefig(base + '.png', dpi=300)
        plt.close(fig)

    print("✔️ Saved integrated spatial summary panels.")

#New basic helpers for loading labels, finding label paths, making colormaps, normalizing values, overlaying mask boundaries, safe filename stems, and reading OME-TIFFs with flexible dimension handling
def load_instance_labels(label_path):
    lab = tiff.imread(label_path)
    lab = np.asarray(lab).astype(np.int32)
    return lab


def find_label_path(out_dir, stem):
    p = os.path.join(out_dir, "cells", f"{stem}_labels.tif")
    return p if os.path.exists(p) else None


def make_channel_linear_cmap(ch, bg="white"):
    """
    Make a white->channel or black->channel colormap for single-marker cell maps.
    """
    channel_hex = get_channel_color(ch)

    if str(bg).lower() == "black":
        start = "#000000"
    else:
        start = "#FFFFFF"

    return LinearSegmentedColormap.from_list(
        f"cellmap_ch{ch}",
        [start, channel_hex]
    )


def normalize_values_for_display(
    vals,
    mode="per_image_percentile",
    pcts=(1, 99),
    display_limits=None
):
    """
    Normalize cell intensity values for display only.

    Important:
      - This does NOT change the measured cell intensities exported to CSV/Excel.
      - "global_percentile" and "manual" should pass display_limits=(lo, hi),
        so all MPs use the same color scale for a given marker.
    """
    vals = np.asarray(vals, dtype=np.float32)

    if vals.size == 0:
        return vals, 0.0, 1.0

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(vals), 0.0, 1.0

    mode = str(mode).strip().lower()

    if display_limits is not None:
        lo, hi = display_limits
        lo = float(lo)
        hi = float(hi)
    elif mode == "per_image_percentile":
        lo, hi = np.percentile(finite, pcts)
    elif mode == "per_image_minmax":
        lo, hi = np.min(finite), np.max(finite)
    else:
        # Safe fallback if global/manual mode was requested but limits were not passed.
        # This keeps the code from crashing, but it is no longer globally comparable.
        lo, hi = np.percentile(finite, pcts)

    if hi <= lo:
        hi = lo + 1e-8

    normed = np.clip((vals - lo) / (hi - lo), 0, 1)
    return normed, float(lo), float(hi)


def build_cell_intensity_display_limits(
    df_cells,
    marker_channels,
    channel_names=None,
    stat="mean",
    mode="global_percentile",
    pcts=(0.5, 99.5),
    manual_limits=None,
    out_dir=None
):
    """
    Build one display scale per marker for cell-intensity maps.

    Returns:
      dict: {channel: (lo, hi)}

    Modes:
      - global_percentile: lo/hi are calculated from all cells across all MPs in this run.
      - manual: lo/hi are taken from CELL_INTENSITY_MANUAL_LIMITS.
      - per_image_*: returns an empty dict because each image is scaled separately.

    This is display-only; it does not affect phenotype thresholds or exported raw cell intensities.
    """
    limits = {}
    rows = []

    if df_cells is None or len(df_cells) == 0:
        return limits

    mode = str(mode).strip().lower()
    manual_limits = manual_limits or {}

    if mode not in ["global_percentile", "manual"]:
        return limits

    for ch in marker_channels:
        channel_name = CHANNEL_NAMES.get(ch, f"ch{ch}") if channel_names is None else channel_names.get(ch, f"ch{ch}")
        stat_col = f"ch{ch}_{stat}"

        if stat_col not in df_cells.columns:
            print(f"⚠️ Missing cell intensity column for display scaling: {stat_col}")
            continue

        vals = (
            df_cells[stat_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=np.float32)
        )

        if vals.size == 0:
            continue

        if mode == "manual":
            lim = manual_limits.get(ch, manual_limits.get(str(ch), None))
            if lim is None:
                raise ValueError(
                    f"CELL_INTENSITY_NORM_MODE='manual' but no display limit was provided for "
                    f"channel {ch} ({channel_name}). Set CELL_INTENSITY_MANUAL_LIMITS[{ch}] = (lo, hi)."
                )
            lo, hi = lim
            source = "manual"
        else:
            lo, hi = np.percentile(vals, pcts)
            source = f"global_percentile_{pcts[0]}_{pcts[1]}"

        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            hi = lo + 1e-8

        limits[int(ch)] = (lo, hi)

        rows.append({
            "channel": int(ch),
            "channel_name": channel_name,
            "stat_col": stat_col,
            "display_scale_mode": mode,
            "display_lo": lo,
            "display_hi": hi,
            "source": source,
            "n_cells_used": int(vals.size),
            "data_min": float(np.min(vals)),
            "data_max": float(np.max(vals)),
            "data_mean": float(np.mean(vals)),
            "data_sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        })

    if rows and out_dir is not None:
        scale_dir = os.path.join(out_dir, "cell_intensity_maps")
        os.makedirs(scale_dir, exist_ok=True)
        scale_path = os.path.join(scale_dir, "cell_intensity_display_scales.csv")
        pd.DataFrame(rows).to_csv(scale_path, index=False)
        print(f"✔️ Exported cell-intensity display scales: {scale_path}")

    return limits


def mask_boundary_overlay(rgb, mask, color=(0, 0, 0)):
    """
    Add colony boundary on top of an RGB image.
    color is RGB in 0..255
    """
    rgb = rgb.copy()
    edge = cv2.Canny((mask > 0).astype(np.uint8) * 255, 50, 150)
    rgb[edge > 0] = color
    return rgb
#helper for standalone phenotype map 
def render_phenotype_label_rgb(labels, df_img, channel_names=None, bg="white"):
    """
    Render each labeled nucleus filled according to phenotype_class.
    """
    if str(bg).lower() == "black":
        rgb = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        boundary_color = (255, 255, 255)
    else:
        rgb = np.ones((labels.shape[0], labels.shape[1], 3), dtype=np.uint8) * 255
        boundary_color = (0, 0, 0)

    # map cell_id -> phenotype_class
    pheno_map = {}
    for _, row in df_img.iterrows():
        pheno_map[int(row["cell_id"])] = str(row["phenotype_class"])

    for cid, pheno in pheno_map.items():
        if cid <= 0:
            continue

        m = (labels == cid)
        if not np.any(m):
            continue

        hex_color = get_phenotype_color(pheno, channel_names=channel_names)
        rgb_color = tuple(int(255 * c) for c in plt.matplotlib.colors.to_rgb(hex_color))
        rgb[m] = rgb_color

    # add label boundaries
    bnd = find_boundaries(labels, mode="thick")
    rgb[bnd] = boundary_color

    return rgb


def save_standalone_cell_phenotype_map(
    stem,
    labels,
    mask,
    df_img,
    out_dir,
    channel_names=None,
    bg="white"
):
    """
    Save one standalone phenotype map with legend outside the map.
    """
    out_dir2 = os.path.join(out_dir, "cell_phenotype_maps")
    os.makedirs(out_dir2, exist_ok=True)

    rgb = render_phenotype_label_rgb(labels, df_img, channel_names=channel_names, bg=bg)

    if str(bg).lower() == "black":
        rgb = mask_boundary_overlay(rgb, mask, color=(255, 255, 255))
    else:
        rgb = mask_boundary_overlay(rgb, mask, color=(0, 0, 0))

    phenotype_classes = sorted(df_img["phenotype_class"].dropna().unique().tolist())
    handles = [
        Patch(facecolor=get_phenotype_color(ph, channel_names=channel_names), edgecolor="none", label=ph)
        for ph in phenotype_classes
    ]

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42])

    ax_map = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    ax_map.imshow(rgb)
    ax_map.set_title(f"{stem} — Cell phenotype map")
    ax_map.axis("off")

    ax_leg.axis("off")
    ax_leg.legend(
        handles=handles,
        loc="upper left",
        fontsize=8,
        frameon=False,
        title="Phenotype class",
        title_fontsize=9
    )

    plt.tight_layout()

    base = os.path.join(out_dir2, f"{stem}_cell_phenotype_map")
    plt.savefig(base + ".svg")
    plt.savefig(base + ".png", dpi=300)
    plt.close()
#helper for single-marker cell intensity maps, coloring each nucleus by its intensity in one marker channel, with normalization and boundary overlay
def render_single_marker_cell_map_rgb(
    labels,
    df_img,
    ch,
    stat="mean",
    bg="white",
    norm_mode="per_image_percentile",
    pcts=(1, 99),
    display_limits=None
):
    """
    Fill each labeled nucleus according to its intensity in one marker channel.
    """
    stat_col = f"ch{ch}_{stat}"
    if stat_col not in df_img.columns:
        raise ValueError(f"Missing column: {stat_col}")

    # image base
    if str(bg).lower() == "black":
        rgb = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        boundary_color = (255, 255, 255)
    else:
        rgb = np.ones((labels.shape[0], labels.shape[1], 3), dtype=np.uint8) * 255
        boundary_color = (0, 0, 0)

    # cell values
    vals = df_img[stat_col].to_numpy(dtype=np.float32)
    normed, lo, hi = normalize_values_for_display(
        vals,
        mode=norm_mode,
        pcts=pcts,
        display_limits=display_limits
    )

    cmap = make_channel_linear_cmap(ch, bg=bg)

    for (_, row), v in zip(df_img.iterrows(), normed):
        cid = int(row["cell_id"])
        m = (labels == cid)
        if not np.any(m):
            continue

        rgb_float = np.array(cmap(float(v))[:3]) * 255
        rgb[m] = rgb_float.astype(np.uint8)

    bnd = find_boundaries(labels, mode="thick")
    rgb[bnd] = boundary_color

    return rgb, cmap, lo, hi

#helper to save single-marker cell intensity maps with legends, one per marker channel, using the above rendering function
def save_single_mp_marker_cell_maps(
    stem,
    labels,
    mask,
    df_img,
    out_dir,
    marker_channels,
    channel_names=None,
    stat="mean",
    bg="white",
    norm_mode="per_image_percentile",
    pcts=(1, 99),
    display_limits_by_channel=None
):
    """
    Save one cell-level intensity map per marker channel for one MP.
    """
    out_dir2 = os.path.join(out_dir, "cell_intensity_maps")
    os.makedirs(out_dir2, exist_ok=True)

    for ch in marker_channels:
        marker_name = channel_names.get(ch, f"ch{ch}") if channel_names is not None else f"ch{ch}"

        display_limits = None
        if display_limits_by_channel is not None:
            display_limits = display_limits_by_channel.get(int(ch))

        rgb, cmap, lo, hi = render_single_marker_cell_map_rgb(
            labels=labels,
            df_img=df_img,
            ch=ch,
            stat=stat,
            bg=bg,
            norm_mode=norm_mode,
            pcts=pcts,
            display_limits=display_limits
        )

        if str(bg).lower() == "black":
            rgb = mask_boundary_overlay(rgb, mask, color=(255, 255, 255))
        else:
            rgb = mask_boundary_overlay(rgb, mask, color=(0, 0, 0))

        fig = plt.figure(figsize=(7.5, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.06])

        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        ax.imshow(rgb)
        ax.set_title(f"{stem} — {marker_name} cell intensity map ({norm_mode})")
        ax.axis("off")

        sm = ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap=cmap)
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label(f"{marker_name} intensity ({stat})")

        plt.tight_layout()

        safe_marker = str(marker_name).replace("/", "_").replace(" ", "_")
        base = os.path.join(out_dir2, f"{stem}_cell_intensity_{safe_marker}")
        plt.savefig(base + ".svg")
        plt.savefig(base + ".png", dpi=300)
        plt.close()
#helper to convert one (R,A) spiderweb grid into a circular rendered scalar image, for visualizing mean marker intensity maps as disks
def grid_channel_to_disk_scalar(z, out_size=700):
    """
    Convert one (R,A) spiderweb grid into a circular rendered scalar image.
    """
    z = np.asarray(z, dtype=np.float32)
    R, A = z.shape

    yy, xx = np.mgrid[0:out_size, 0:out_size]
    cx = (out_size - 1) / 2.0
    cy = (out_size - 1) / 2.0

    xn = (xx - cx) / cx
    yn = (yy - cy) / cy

    rr = np.sqrt(xn**2 + yn**2)
    theta = np.arctan2(-yn, xn)  # flip y for image coords
    theta = np.mod(theta, 2 * np.pi)

    inside = rr <= 1.0

    r_idx = np.clip((rr * (R - 1)).round().astype(int), 0, R - 1)
    a_idx = np.clip((theta / (2 * np.pi) * A).astype(int), 0, A - 1)

    img = np.zeros((out_size, out_size), dtype=np.float32)
    img[inside] = z[r_idx[inside], a_idx[inside]]

    return img, inside.astype(np.uint8)

#helper to save grouped mean marker maps rendered as full disks, one per marker channel, using the above grid-to-disk rendering function
def save_grouped_disk_marker_maps(
    out_dir,
    channel_names=None,
    marker_channels=None,
    bg="white"
):
    """
    Save grouped mean marker maps rendered as full disks.
    """
    if marker_channels is None:
        marker_channels = PHENOTYPE_MARKER_CHANNELS

    mean_grid, sd_grid, n_grids = load_and_average_saved_grids(out_dir)
    if mean_grid is None or n_grids == 0:
        print("⚠️ No grouped grids available for grouped disk marker maps.")
        return

    out_dir2 = os.path.join(out_dir, "grouped_cell_intensity_maps")
    os.makedirs(out_dir2, exist_ok=True)

    for ch in marker_channels:
        marker_name = channel_names.get(ch, f"ch{ch}") if channel_names is not None else f"ch{ch}"
        cmap = make_channel_linear_cmap(ch, bg=bg)

        img, inside = grid_channel_to_disk_scalar(mean_grid[ch], out_size=700)

        if str(bg).lower() == "black":
            rgb_base = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            boundary_color = (255, 255, 255)
        else:
            rgb_base = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            boundary_color = (0, 0, 0)

        rgb = rgb_base.copy()
        rgb_vals = (np.array(cmap(img)[..., :3]) * 255).astype(np.uint8)
        rgb[inside > 0] = rgb_vals[inside > 0]

        edge = cv2.Canny((inside > 0).astype(np.uint8) * 255, 50, 150)
        rgb[edge > 0] = boundary_color

        fig = plt.figure(figsize=(6.5, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.06])

        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        ax.imshow(rgb)
        ax.set_title(f"Grouped marker map — {marker_name} (n={n_grids})")
        ax.axis("off")

        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label(f"{marker_name} mean normalized intensity")

        plt.tight_layout()

        safe_marker = str(marker_name).replace("/", "_").replace(" ", "_")
        base = os.path.join(out_dir2, f"grouped_disk_marker_map_{safe_marker}_n{n_grids}")
        plt.savefig(base + ".svg")
        plt.savefig(base + ".png", dpi=300)
        plt.close()

    print("✔️ Saved grouped disk marker maps.")

#helper for safe filename stem (removing problematic chars)
def safe_stem(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in stem)

def read_ome_as_CYX(tif_path, z_mode="max"):
    """
    Read OME-TIFF and return (C,Y,X) explicitly.
    Handles (C,Y,X), (C,Z,Y,X), (Z,C,Y,X), (Y,X), etc.
    z_mode: "max", "mean", "mid", or an int z index
    """
    a = np.asarray(tiff.imread(tif_path))

    # (Y,X)
    if a.ndim == 2:
        return a[None, ...]

    # (C,Y,X) or (Y,X,C)
    if a.ndim == 3:
        if a.shape[0] <= 8:
            return a
        if a.shape[2] <= 8:
            return np.transpose(a, (2, 0, 1))
        return a

    # (C,Z,Y,X) or (Z,C,Y,X) common cases
    if a.ndim == 4:
        # guess (C,Z,Y,X)
        if a.shape[0] <= 8 and a.shape[1] > 1:
            C, Z, Y, X = a.shape
            if z_mode == "max":
                return a.max(axis=1)
            if z_mode == "mean":
                return a.mean(axis=1)
            if z_mode == "mid":
                return a[:, Z // 2]
            if isinstance(z_mode, int):
                return a[:, int(z_mode)]
            raise ValueError("z_mode must be 'max','mean','mid', or int")

        # guess (Z,C,Y,X)
        if a.shape[1] <= 8 and a.shape[0] > 1:
            Z, C, Y, X = a.shape
            if z_mode == "max":
                return a.max(axis=0)
            if z_mode == "mean":
                return a.mean(axis=0)
            if z_mode == "mid":
                return a[Z // 2, :]
            if isinstance(z_mode, int):
                return a[int(z_mode), :]
            raise ValueError("z_mode must be 'max','mean','mid', or int")

        # fallback: take first frame-like dim then reuse existing logic
        return read_ome_as_CYX_from_array(a[0], z_mode=z_mode)

    # 5D+ (T,Z,C,Y,X) etc: peel leading dims
    while a.ndim > 4:
        a = a[0]
    return read_ome_as_CYX_from_array(a, z_mode=z_mode)


def read_ome_as_CYX_from_array(a, z_mode="max"):
    # helper for the fallback; keeps code simple
    # reuse the same logic by writing to temp-like path is annoying; so just copy
    a = np.asarray(a)
    if a.ndim == 2:
        return a[None, ...]
    if a.ndim == 3:
        if a.shape[0] <= 8:
            return a
        if a.shape[2] <= 8:
            return np.transpose(a, (2, 0, 1))
        return a
    if a.ndim == 4:
        if a.shape[0] <= 8 and a.shape[1] > 1:
            C, Z, Y, X = a.shape
            if z_mode == "max":
                return a.max(axis=1)
            if z_mode == "mean":
                return a.mean(axis=1)
            if z_mode == "mid":
                return a[:, Z // 2]
            if isinstance(z_mode, int):
                return a[:, int(z_mode)]
        if a.shape[1] <= 8 and a.shape[0] > 1:
            Z, C, Y, X = a.shape
            if z_mode == "max":
                return a.max(axis=0)
            if z_mode == "mean":
                return a.mean(axis=0)
            if z_mode == "mid":
                return a[Z // 2, :]
            if isinstance(z_mode, int):
                return a[int(z_mode), :]
    raise ValueError(f"Unsupported array shape for OME: {a.shape}")

def normalize01(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)
#new normalization helper to preserve relative differences but scale to max=1 for better cross-sample comparison and visualization
def normalize_to_curve_max(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mx = float(np.max(x))
    if mx < eps:
        return np.zeros_like(x, dtype=np.float32), mx
    return x / (mx + eps), mx
# Map each nucleus centroid to r_frac and theta
def _theta_to_bin(theta, n_angles):
    # theta in [0, 2pi)
    return int(np.floor(theta / (2*np.pi) * n_angles)) % n_angles

def cell_polar_coords(x, y, cx, cy, angles, boundary_r):
    dx = x - cx
    dy = y - cy
    theta = float(np.arctan2(dy, dx))
    if theta < 0:
        theta += 2*np.pi

    r = float(np.hypot(dx, dy))

    ai = _theta_to_bin(theta, len(angles))
    rb = float(boundary_r[ai])
    r_frac = r / (rb + 1e-8)
    r_frac = float(np.clip(r_frac, 0.0, 1.5))

    return r_frac, theta, ai
#Nuclei segmentation helper 
def estimate_single_nucleus_area(df_obj,
                                 min_area_for_ref=60,
                                 max_area_quantile=0.60,
                                 min_solidity=0.90,
                                 max_eccentricity=0.85):
    """
    Estimate typical single-nucleus area from cleaner, rounder objects.
    Returns median reference area in pixels.
    """
    if df_obj is None or len(df_obj) == 0:
        return np.nan

    df = df_obj.copy()

    # keep only plausible, cleaner objects for reference
    area_cap = df["area_px"].quantile(max_area_quantile) if len(df) > 5 else df["area_px"].max()

    ref = df[
        (df["area_px"] >= min_area_for_ref) &
        (df["area_px"] <= area_cap) &
        (df["solidity"] >= min_solidity) &
        (df["eccentricity"] <= max_eccentricity)
    ].copy()

    # fallback if too strict
    if len(ref) < 8:
        ref = df[df["area_px"] >= min_area_for_ref].copy()

    if len(ref) == 0:
        return np.nan

    return float(ref["area_px"].median())
def annotate_clumps(df_obj, single_area,
                    clump_area_factor=1.8,
                    low_solidity=0.88,
                    high_eccentricity=0.92):
    """
    Annotate each segmented object as likely single nucleus or clump.
    Adds:
      - is_clump
      - est_nuclei_by_area
    """
    if df_obj is None or len(df_obj) == 0:
        return df_obj

    df = df_obj.copy()

    if not np.isfinite(single_area) or single_area <= 0:
        df["is_clump"] = False
        df["est_nuclei_by_area"] = 1
        return df

    area_ratio = df["area_px"] / float(single_area)

    df["is_clump"] = (
        (area_ratio >= clump_area_factor) |
        (df["solidity"] < low_solidity) |
        (df["eccentricity"] > high_eccentricity)
    )

    df["est_nuclei_by_area"] = np.maximum(1, np.rint(area_ratio)).astype(int)

    return df

def nuclei_object_table(labels):
    """
    Build per-object morphology table from labeled nuclei image.
    Returns one row per segmented object.
    """
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return pd.DataFrame(columns=[
            "cell_id", "x", "y", "area_px",
            "eccentricity", "solidity", "extent",
            "equivalent_diameter", "major_axis_length", "minor_axis_length"
        ])

    props = regionprops_table(
        labels,
        properties=(
            "label",
            "centroid",
            "area",
            "eccentricity",
            "solidity",
            "extent",
            "equivalent_diameter_area",
            "axis_major_length",
            "axis_minor_length"
        )
    )

    df = pd.DataFrame(props).rename(columns={
        "label": "cell_id",
        "centroid-1": "x",
        "centroid-0": "y",
        "area": "area_px",
        "equivalent_diameter_area": "equivalent_diameter",
        "axis_major_length": "major_axis_length",
        "axis_minor_length": "minor_axis_length",
    })

    return df

def segment_nuclei(dapi, mask, gauss_sigma=0.8, min_area=35, max_area=300,
                   sigma_bg=12, blockSize=35, C=-4, min_distance=3):
    """
    Improved nuclei segmentation for dense DAPI carpets.
    Returns labeled nuclei image (0 background, 1..N nuclei).
    """
    mask = (mask > 0).astype(np.uint8)
    d = dapi.astype(np.float32)

    # 1) smooth
    if gauss_sigma and gauss_sigma > 0:
        d_s = cv2.GaussianBlur(d, (0, 0), gauss_sigma)
    else:
        d_s = d.copy()

    # 2) flat-field correction
    bg = cv2.GaussianBlur(d_s, (0, 0), sigma_bg)
    d_corr = d_s / (bg + 1e-6)
    d_corr[mask == 0] = 0

    vals = d_corr[mask > 0]
    if vals.size < 50:
        return np.zeros_like(mask, dtype=np.int32)

    # 3) robust normalize to 8-bit
    lo, hi = np.percentile(vals, [1, 99.5])
    d_norm = np.clip((d_corr - lo) / (hi - lo + 1e-8), 0, 1)
    d8 = (d_norm * 255).astype(np.uint8)

    # 4) adaptive threshold
    bw8 = cv2.adaptiveThreshold(
        d8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize,
        C
    )
    bw = (bw8 > 0) & (mask > 0)

    # 5) cleanup
    bw = ndi.binary_opening(bw, structure=np.ones((3, 3))) 
    bw = ndi.binary_closing(bw, structure=np.ones((3, 3)))
    bw = ndi.binary_fill_holes(bw)


    if bw.sum() == 0:
        return np.zeros_like(mask, dtype=np.int32)

    # 6) distance transform
    dist = ndi.distance_transform_edt(bw)

    # 7) local maxima seeds
    coords = peak_local_max(
        dist,
        min_distance=min_distance,
        labels=bw,
        exclude_border=False
    )

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        cc, _ = ndi.label(bw)
        labels = cc.astype(np.int32)
    else:
        markers, _ = ndi.label(markers > 0)
        labels = watershed(-dist, markers, mask=bw)

    # 8) area filter
    areas = np.bincount(labels.ravel())
    keep = (areas >= min_area) & (areas <= max_area)
    keep[0] = False
    labels[~keep[labels]] = 0

    # relabel
    labels, _ = ndi.label(labels > 0)
    return labels.astype(np.int32)


# -------------------------
# StarDist instance segmentation helpers
# -------------------------
def crop_to_mask_bbox(img2d, mask, pad=20):
    """
    Crop a 2D image and mask to the colony bounding box.
    This keeps StarDist much faster than running on the full field.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return img2d, mask_u8, (0, 0)

    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(mask_u8.shape[1], int(xs.max()) + int(pad) + 1)
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(mask_u8.shape[0], int(ys.max()) + int(pad) + 1)

    return img2d[y0:y1, x0:x1], mask_u8[y0:y1, x0:x1], (y0, x0)


def relabel_instance_mask(labels):
    """
    Reindex any instance mask to 1..N while preserving separated objects.
    """
    labels = np.asarray(labels).astype(np.int32)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids != 0]

    if len(unique_ids) == 0:
        return np.zeros_like(labels, dtype=np.int32)

    new_labels = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        new_labels[labels == old_id] = new_id

    return new_labels


def filter_instance_area(labels, min_area=35, max_area=300):
    """
    Remove segmented instances outside the expected nucleus area range.
    """
    labels = np.asarray(labels).astype(np.int32)
    if labels.max() == 0:
        return labels

    areas = np.bincount(labels.ravel())
    keep = np.ones_like(areas, dtype=bool)
    keep[0] = False

    if min_area is not None:
        keep &= areas >= int(min_area)
    if max_area is not None:
        keep &= areas <= int(max_area)

    labels[~keep[labels]] = 0
    return relabel_instance_mask(labels)


STARDIST_MODEL = None

def get_stardist_model():
    """
    Load the StarDist model once, then reuse it for all images.
    """
    global STARDIST_MODEL

    if not STARDIST_AVAILABLE:
        raise ImportError(
            "StarDist is not installed. Install it in your venv with: pip install stardist"
        )

    if STARDIST_MODEL is None:
        print(f"[StarDist] loading pretrained model: {STARDIST_MODEL_NAME}")
        STARDIST_MODEL = StarDist2D.from_pretrained(STARDIST_MODEL_NAME)
        print("[StarDist] model loaded")

    return STARDIST_MODEL


def segment_nuclei_stardist(
    dapi,
    mask,
    prob_thresh=None,
    nms_thresh=None,
    min_area=35,
    max_area=300,
    bbox_pad=20
):
    """
    StarDist-based nuclei segmentation for a single 2D DAPI image.
    Includes explicit debug prints to confirm thresholds are actually passed.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    vals = dapi[mask_u8 > 0]

    if vals.size < 50:
        return np.zeros_like(mask_u8, dtype=np.int32)

    d_crop, m_crop, (y0, x0) = crop_to_mask_bbox(
        dapi.astype(np.float32),
        mask_u8,
        pad=bbox_pad
    )

    vals = d_crop[m_crop > 0]
    if vals.size < 50:
        return np.zeros_like(mask_u8, dtype=np.int32)

    # Keep this preprocessing close to the version that worked for 500 µm circles.
    lo, hi = np.percentile(vals, [1, 99.8])
    d_norm = np.clip((d_crop - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
    d_norm[m_crop == 0] = 0

    model = get_stardist_model()

    if prob_thresh is None:
        prob_thresh = STARDIST_PROB_THRESH
    if nms_thresh is None:
        nms_thresh = STARDIST_NMS_THRESH

    kwargs = {
        "prob_thresh": float(prob_thresh),
        "nms_thresh": float(nms_thresh),
    }

    print("\n" + "=" * 70)
    print("[StarDist DEBUG]")
    print(f"input crop: {d_norm.shape}, dtype: {d_norm.dtype}")
    print(f"FORCED prob_thresh = {kwargs['prob_thresh']}")
    print(f"FORCED nms_thresh  = {kwargs['nms_thresh']}")
    print(f"area filter: {min_area} to {max_area}")
    print("=" * 70 + "\n")

    labels_crop, details = model.predict_instances(d_norm, **kwargs)
    labels_crop = np.asarray(labels_crop).astype(np.int32)

    raw_count = int(len(np.unique(labels_crop)) - 1)
    print(f"[StarDist DEBUG] Raw instances before mask/filter: {raw_count}")

    labels_crop[m_crop == 0] = 0

    after_mask_count = int(len(np.unique(labels_crop)) - 1)
    print(f"[StarDist DEBUG] Instances after MP mask: {after_mask_count}")

    labels_crop = filter_instance_area(
        labels_crop,
        min_area=min_area,
        max_area=max_area
    )

    after_area_count = int(labels_crop.max())
    print(f"[StarDist DEBUG] Instances after area filter: {after_area_count}")

    labels = np.zeros_like(mask_u8, dtype=np.int32)
    labels[y0:y0 + labels_crop.shape[0], x0:x0 + labels_crop.shape[1]] = labels_crop
    labels[mask_u8 == 0] = 0

    labels = relabel_instance_mask(labels)

    final_count = int(labels.max())
    print(f"[StarDist DEBUG] Final nuclei count: {final_count}")

    return labels


def segment_nuclei_dense_watershed(
    dapi,
    mask,
    min_area=20,
    max_area=450,
    gauss_sigma=0.6,
    sigma_bg=25,
    blockSize=31,
    C=-5,
    min_distance=4
):
    """
    Dense DAPI watershed fallback for very packed micropatterns.

    This is the fast fallback for Shapes V2 when StarDist only detects
    high-contrast edge nuclei and misses the packed interior.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    d = dapi.astype(np.float32)

    if np.sum(mask_u8) < 50:
        return np.zeros_like(mask_u8, dtype=np.int32)

    # Mild smoothing
    d_s = cv2.GaussianBlur(d, (0, 0), gauss_sigma)

    # Flat-field correction
    bg = cv2.GaussianBlur(d_s, (0, 0), sigma_bg)
    d_corr = d_s / (bg + 1e-6)
    d_corr[mask_u8 == 0] = 0

    vals = d_corr[mask_u8 > 0]
    if vals.size < 50:
        return np.zeros_like(mask_u8, dtype=np.int32)

    # Robust normalize to 8-bit
    lo, hi = np.percentile(vals, [1, 99.5])
    d_norm = np.clip((d_corr - lo) / (hi - lo + 1e-8), 0, 1)
    d8 = (d_norm * 255).astype(np.uint8)

    # Adaptive threshold works better than StarDist in very dense DAPI carpets
    bw8 = cv2.adaptiveThreshold(
        d8,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        int(blockSize),
        float(C)
    )

    bw = (bw8 > 0) & (mask_u8 > 0)

    # Clean but do not over-open, or you delete real nuclei
    bw = ndi.binary_opening(bw, structure=np.ones((2, 2)))
    bw = ndi.binary_fill_holes(bw)

    if bw.sum() == 0:
        return np.zeros_like(mask_u8, dtype=np.int32)

    # Distance transform
    dist = ndi.distance_transform_edt(bw)

    # Seeds
    coords = peak_local_max(
        dist,
        min_distance=int(min_distance),
        labels=bw,
        exclude_border=False
    )

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        labels, _ = ndi.label(bw)
    else:
        markers, _ = ndi.label(markers > 0)
        labels = watershed(-dist, markers, mask=bw)

    labels = labels.astype(np.int32)

    # Area filter
    areas = np.bincount(labels.ravel())
    keep = np.ones_like(areas, dtype=bool)
    keep[0] = False
    keep &= areas >= int(min_area)
    keep &= areas <= int(max_area)

    labels[~keep[labels]] = 0
    labels = relabel_instance_mask(labels)

    print(f"[Dense watershed fallback] Final nuclei count: {int(labels.max())}")

    return labels


def segment_nuclei_hybrid(dapi, mask):
    """
    Run StarDist first. If the count is suspiciously low, use dense watershed fallback.
    """
    labels_sd = segment_nuclei_stardist(
        dapi=dapi,
        mask=mask,
        prob_thresh=STARDIST_PROB_THRESH,
        nms_thresh=STARDIST_NMS_THRESH,
        min_area=STARDIST_MIN_SIZE,
        max_area=STARDIST_MAX_SIZE,
        bbox_pad=STARDIST_BBOX_PAD
    )

    n_sd = int(labels_sd.max())
    print(f"[HYBRID] StarDist count = {n_sd}")

    if n_sd >= HYBRID_MIN_ACCEPTABLE_NUCLEI:
        print("[HYBRID] Keeping StarDist segmentation.")
        return labels_sd

    print(
        f"[HYBRID] StarDist count below {HYBRID_MIN_ACCEPTABLE_NUCLEI}. "
        "Switching to dense watershed fallback."
    )

    labels_ws = segment_nuclei_dense_watershed(
        dapi=dapi,
        mask=mask,
        min_area=DENSE_WS_MIN_AREA,
        max_area=DENSE_WS_MAX_AREA,
        gauss_sigma=DENSE_WS_GAUSS_SIGMA,
        sigma_bg=DENSE_WS_BG_SIGMA,
        blockSize=DENSE_WS_BLOCK_SIZE,
        C=DENSE_WS_C,
        min_distance=DENSE_WS_MIN_DISTANCE
    )

    return labels_ws

#extract per cell table + overlay option helper 
def extract_cell_table(labels, img, stem, cx, cy, angles, boundary_r,
                       n_sectors=12, ring_edges=(0.33, 0.67)):
    """
    labels: (Y,X) int
    img: (C,Y,X)
    returns df with one row per segmented object + clump-aware fields
    """
    re1, re2 = ring_edges
    C = img.shape[0]

    df_obj = nuclei_object_table(labels)
    if len(df_obj) == 0:
        return pd.DataFrame()

    single_area = estimate_single_nucleus_area(df_obj)
    df_obj = annotate_clumps(df_obj, single_area)

    rows = []

    for _, rr in df_obj.iterrows():
        cid = int(rr["cell_id"])
        x = float(rr["x"])
        y = float(rr["y"])
        m = (labels == cid)

        r_frac, theta, ai = cell_polar_coords(x, y, cx, cy, angles, boundary_r)

        if r_frac < re1:
            ring_bin = 0
        elif r_frac < re2:
            ring_bin = 1
        else:
            ring_bin = 2

        sector_bin = int(np.floor(theta / (2*np.pi) * n_sectors)) % n_sectors

        row = {
            "image": stem,
            "cell_id": cid,
            "x": x,
            "y": y,
            "r_frac": float(r_frac),
            "theta_rad": float(theta),
            "theta_deg": float(theta * 180 / np.pi),
            "ring_bin": int(ring_bin),
            "sector_bin": int(sector_bin),

            # morphology
            "area_px": int(rr["area_px"]),
            "eccentricity": float(rr["eccentricity"]),
            "solidity": float(rr["solidity"]),
            "extent": float(rr["extent"]),
            "equivalent_diameter": float(rr["equivalent_diameter"]),
            "major_axis_length": float(rr["major_axis_length"]),
            "minor_axis_length": float(rr["minor_axis_length"]),

            # clump-aware fields
            "single_nucleus_area_ref": float(single_area) if np.isfinite(single_area) else np.nan,
            "is_clump": bool(rr["is_clump"]),
            "est_nuclei_by_area": int(rr["est_nuclei_by_area"]),
        }

        for ch in range(C):
            pix = img[ch][m].astype(np.float32)
            row[f"ch{ch}_mean"] = float(np.mean(pix))
            row[f"ch{ch}_p90"] = float(np.percentile(pix, 90))
            row[f"ch{ch}_sum"] = float(np.sum(pix))

        rows.append(row)

    return pd.DataFrame(rows)

def nuclei_overlay_boundaries(dapi, labels, mask):
    """
    DAPI grayscale + full boundaries for EACH labeled nucleus + mask boundary.
    Green = nucleus instance boundaries
    Red = colony/mask boundary
    """
    from skimage.segmentation import find_boundaries

    d = dapi.astype(np.float32)

    if np.any(mask):
        lo, hi = np.percentile(d[mask > 0], [1, 99.5])
    else:
        lo, hi = np.percentile(d, [1, 99.5])

    d = np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)
    base = (d * 255).astype(np.uint8)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # IMPORTANT: boundaries from full label image, not (labels > 0)
    bnd = find_boundaries(labels, mode="thick")
    rgb[bnd] = (0, 255, 0)   # green

    # mask boundary in red
    mask_edge = cv2.Canny((mask > 0).astype(np.uint8) * 255, 50, 150)
    rgb[mask_edge > 0] = (0, 0, 255)

    return rgb


def nuclei_overlay_colored_instances(dapi, labels, mask, alpha=0.45, seed=7):
    """
    DAPI grayscale + each labeled nucleus filled with a different color.
    Adds white boundaries between cells and red colony boundary.
    """
    from skimage.segmentation import find_boundaries

    d = dapi.astype(np.float32)

    if np.any(mask):
        lo, hi = np.percentile(d[mask > 0], [1, 99.5])
    else:
        lo, hi = np.percentile(d, [1, 99.5])

    d = np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)

    # grayscale base
    base = (d * 255).astype(np.uint8)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR).astype(np.float32)

    rng = np.random.default_rng(seed)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids != 0]

    # random but reproducible colors
    color_table = {}
    for cid in unique_ids:
        color_table[cid] = rng.integers(60, 256, size=3)  # brighter colors

    overlay = rgb.copy()

    for cid in unique_ids:
        m = labels == cid
        color = color_table[cid].astype(np.float32)
        overlay[m] = (1 - alpha) * overlay[m] + alpha * color

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # boundaries between touching cells
    bnd = find_boundaries(labels, mode="thick")
    overlay[bnd] = (255, 255, 255)   # white boundaries

    # mask boundary
    mask_edge = cv2.Canny((mask > 0).astype(np.uint8) * 255, 50, 150)
    overlay[mask_edge > 0] = (0, 0, 255)

    return overlay


def nuclei_overlay_png(dapi, labels, mask):
    """
    Backward-compatible default QC overlay.
    """
    return nuclei_overlay_boundaries(dapi, labels, mask)
#------------------------------------------------------ Deliverable 2 helper functions end
#orthogonal profile figure saving helper
def save_orthogonal_profiles_figure(stem, out_dir, r_fracs, grid, theta0, n_angles):
    """
    Save one figure with major/minor axis radial profiles for all channels.
    """
    C = grid.shape[0]

    plt.figure(figsize=(8, 5))
    for ch in range(C):
        pmaj, pmin = orthogonal_profiles_from_grid(
            grid[ch],
            theta0,
            n_angles,
            half_width_bins=3
        )

        color = get_channel_color(ch)

        plt.plot(
            r_fracs,
            pmaj,
            linewidth=2.2,
            color=color,
            label=f"ch{ch} major"
        )

        plt.plot(
            r_fracs,
            pmin,
            linewidth=2.2,
            linestyle="--",
            color=color,
            alpha=0.75,
            label=f"ch{ch} minor"
        )
    plt.title(f"{stem} — Orthogonal major/minor profiles")
    plt.xlabel("Radius fraction (0→1)")
    plt.ylabel("Normalized intensity")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "profiles", f"{stem}_orthogonal_profiles.svg"))
    plt.close()
#save cleaner per chanel topography maps 
def save_topography_maps(stem, out_dir, grid, r_fracs):
    """
    Save one topography map per channel from the spiderweb grid.
    """
    C = grid.shape[0]

    for ch in range(C):
        plt.figure(figsize=(8, 4))
        plt.imshow(
            grid[ch],
            aspect="auto",
            origin="lower",
            extent=[0, 360, float(r_fracs[0]), float(r_fracs[-1])],
            vmin=0,
            vmax=1,
            cmap=get_channel_cmap(ch)
        )
        plt.colorbar(label="Normalized intensity")
        plt.title(f"{stem} — Channel {ch} topography map")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Radius fraction")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heatmaps", f"{stem}_topography_ch{ch}.svg"))
        plt.close()
#polar polography helper 
def save_polar_topography_maps(stem, out_dir, angles, r_fracs, grid):
    """
    Save one polar topography map per channel from the spiderweb grid.
    Mesh is rasterized to keep files editable but Inkscape-friendly.
    """
    C = grid.shape[0]

    if len(angles) > 1:
        dtheta = float(np.mean(np.diff(angles)))
    else:
        dtheta = 2 * np.pi

    theta_edges = np.concatenate([angles, [angles[-1] + dtheta]])

    if len(r_fracs) > 1:
        dr = float(np.mean(np.diff(r_fracs)))
    else:
        dr = 1.0

    r_edges = np.concatenate([
        [max(0.0, r_fracs[0] - dr / 2)],
        (r_fracs[:-1] + r_fracs[1:]) / 2,
        [r_fracs[-1] + dr / 2]
    ])

    for ch in range(C):
        z = grid[ch]

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="polar")

        pcm = ax.pcolormesh(
            theta_edges,
            r_edges,
            z,
            shading="auto",
            vmin=0,
            vmax=1,
            cmap=get_channel_cmap(ch),
            rasterized=True
        )

        ax.set_title(f"{stem} — Polar topography ch{ch}", pad=20)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 1.0)
        ax.grid(alpha=0.25)

        cbar = plt.colorbar(pcm, ax=ax, pad=0.12)
        cbar.set_label("Normalized intensity")

        plt.tight_layout()

        base = os.path.join(out_dir, "topography_maps", f"{stem}_polar_topography_ch{ch}")
        plt.savefig(base + ".pdf", bbox_inches="tight", dpi=300)
        plt.savefig(base + ".png", bbox_inches="tight", dpi=600)

        plt.close()
# -------------------------
# Masking
# -------------------------
def keep_largest_component(mask_u8):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8
    # stats[:, cv2.CC_STAT_AREA] includes background at index 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8)

def make_mask(img_CYX, mode="composite", dapi_channel=0,
              blur=1.5,
              thresh="percentile", thresh_pct=40,
              morph_close=11, morph_open=5,
              bg_radius=35,
              dilate_px=4,
              sauvola_window=61,
              sauvola_k=0.2,
              tophat_radius=9,
              min_obj_size=2000):
    """
    Better MP mask:
      - mode="composite" or "dapi": current behavior
      - mode="dapi_sauvola": DAPI-focused preprocessing inspired by nuclei pipeline

    Returns:
      binary colony mask (uint8, 0/1)
    """
    from skimage.filters import threshold_sauvola
    from skimage.morphology import white_tophat, disk, remove_small_objects
    from scipy import ndimage as ndi

    C, H, W = img_CYX.shape

    # -----------------------------
    # MODE 1 / 2: original behavior
    # -----------------------------
    if mode in ["dapi", "composite"]:
        # 1) base image
        if mode == "dapi":
            base = img_CYX[int(dapi_channel)].astype(np.float32)
        else:  # composite
            chans = [img_CYX[c].astype(np.float32) for c in range(C)]
            chans = [normalize01(ch) for ch in chans]
            base = np.max(np.stack(chans, axis=0), axis=0).astype(np.float32)

        # 2) background subtraction
        k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_radius, bg_radius))
        bg = cv2.morphologyEx(base, cv2.MORPH_OPEN, k_bg)
        base = base - bg
        base = normalize01(base)

        # 3) blur
        base = cv2.GaussianBlur(base, (0, 0), float(blur))

        # 4) threshold
        if thresh == "otsu":
            base8 = (np.clip(base, 0, 1) * 255).astype(np.uint8)
            _, mask = cv2.threshold(base8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = mask.astype(np.uint8)
        elif thresh == "percentile":
            t = np.percentile(base, float(thresh_pct))
            mask = (base > t).astype(np.uint8)
        else:
            raise ValueError("thresh must be 'otsu' or 'percentile'")

    # ---------------------------------------
    # MODE 3: DAPI + tophat + Sauvola masking
    # ---------------------------------------
    elif mode == "dapi_sauvola":
        base = img_CYX[int(dapi_channel)].astype(np.float32)

        # robust percentile normalization
        p1, p99 = np.percentile(base, (1, 99))
        base = np.clip((base - p1) / (p99 - p1 + 1e-8), 0, 1)

        # mild blur
        base = cv2.GaussianBlur(base, (0, 0), float(blur))

        # white tophat enhances nuclei against slow background
        base_tophat = white_tophat(base, footprint=disk(tophat_radius))
        base_tophat = normalize01(base_tophat)

        # Sauvola local threshold
        local_thr = threshold_sauvola(base_tophat, window_size=sauvola_window, k=sauvola_k)
        nuclei_mask = base_tophat > local_thr

        # morphology cleanup
        nuclei_mask = remove_small_objects(nuclei_mask, min_size=40)
        nuclei_mask = ndi.binary_fill_holes(nuclei_mask)
        nuclei_mask = cv2.morphologyEx(nuclei_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close)),
                                       iterations=1)

        # turn nuclei foreground into colony-support mask
        # stronger closing merges dense nuclei carpet into one colony region
        colony_mask = cv2.morphologyEx(nuclei_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                                       iterations=2)

        colony_mask = ndi.binary_fill_holes(colony_mask > 0)
        colony_mask = remove_small_objects(colony_mask, min_size=min_obj_size)
        mask = colony_mask.astype(np.uint8)

    else:
        raise ValueError("mode must be 'dapi', 'composite', or 'dapi_sauvola'")

    # -----------------------------
    # shared cleanup for all modes
    # -----------------------------
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))

    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k_close, iterations=2)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,  k_open,  iterations=1)

    # fill holes
    mask_flood = mask.copy()
    flood = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(mask_flood, flood, (0, 0), 1)
    holes = (mask_flood == 0).astype(np.uint8)
    mask = np.clip(mask + holes, 0, 1).astype(np.uint8)

    # keep largest component
    mask = keep_largest_component(mask)

    # optional dilation
    if dilate_px and dilate_px > 0:
        k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        mask = cv2.dilate(mask, k_d, iterations=1)

    return mask

def load_mask(mask_path, target_shape):
    m = tiff.imread(mask_path)
    m = np.asarray(m)
    if m.ndim > 2:
        m = m.squeeze()
    if m.shape != target_shape:
        raise ValueError(f"Mask shape {m.shape} != image shape {target_shape}")
    return (m > 0).astype(np.uint8)

def mask_centroid(mask_u8):
    ys, xs = np.nonzero(mask_u8 > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())  # cx, cy

# -------------------------
# Spiderweb sampling
# -------------------------
def ray_boundary_distance(mask_u8, cx, cy, ang, step=1.0, max_steps=20000):
    h, w = mask_u8.shape
    dx, dy = np.cos(ang), np.sin(ang)
    last_in = 0.0
    for i in range(1, max_steps):
        r = i * step
        x = cx + dx * r
        y = cy + dy * r
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            break
        if mask_u8[yi, xi] == 0:
            break
        last_in = r
    return last_in

#temporary sanity check for channels 
def save_raw_channel_qc(img, stem, out_dir):
    C = img.shape[0]
    fig, axes = plt.subplots(1, C, figsize=(4*C, 4))

    if C == 1:
        axes = [axes]

    for ch in range(C):
        ax = axes[ch]
        ax.imshow(normalize01(img[ch]), cmap="gray")
        ax.set_title(f"Raw Python ch{ch}\nHuman ch{ch+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "debug", f"{stem}_RAW_CHANNEL_QC.png"), dpi=200)
    plt.close()

#Note: n_angles: number of radial spokes, n_r: number of points along each spoke, r_min_frac: minimum radius fraction to sample (0.0 = from center, >0.0 = start sampling from that fraction of the radius), sigma_r: optional smoothing along the radial direction (in units of r bins).
#n_angles can be changed to less or more radial spokes as needed 
def spiderweb_sample(img_CYX, mask_u8, n_angles=240, n_r=200, r_min_frac=0.0, sigma_r=2.5):
    """
    Returns:
      angles (A,), r_fracs (R,),
      grid (C,R,A) normalized per channel 0-1,
      boundary_r_px (A,), centroid (cx,cy)
    """
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    C, H, W = img_CYX.shape

    ctr = mask_centroid(mask_u8)
    if ctr is None:
        raise ValueError("Empty mask — cannot sample.")
    cx, cy = ctr

    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False).astype(np.float32)
    boundary_r = np.array([ray_boundary_distance(mask_u8, cx, cy, a) for a in angles], dtype=np.float32)
    boundary_r = np.maximum(boundary_r, 1.0)

    r_fracs = np.linspace(r_min_frac, 1.0, n_r).astype(np.float32)

    rr = r_fracs[:, None] * boundary_r[None, :]
    xx = cx + rr * np.cos(angles)[None, :]
    yy = cy + rr * np.sin(angles)[None, :]

    coords = np.vstack([yy.reshape(-1), xx.reshape(-1)])

    grid = np.zeros((C, n_r, n_angles), dtype=np.float32)
    for ch in range(C):
        sampled = map_coordinates(img_CYX[ch].astype(np.float32), coords, order=1, mode="nearest")
        sampled = sampled.reshape(n_r, n_angles)
        sampled = normalize01(sampled)
        if sigma_r and sigma_r > 0:
            sampled = gaussian_filter1d(sampled, sigma=sigma_r, axis=0)
        grid[ch] = sampled

    return angles, r_fracs, grid, boundary_r, cx, cy

#note: this is designed to be a visual for the spiderweb sampling, n_spokes are the visual one we can see on the overlay, 
# while the n_angles are the real sampling spokes used for the grid. 
# The overlay can be sparser for clarity, while the grid can be denser for analysis.
def debug_overlay(img_CYX, mask_u8, cx, cy, boundary_r, n_spokes=60, ring_fracs=(0.25, 0.5, 0.75, 1.0)):
    """
    Marker-only spiderweb overlay.
    Uses channels 1, 2, 3 when available, excluding DAPI ch0.
    Draws the spiderweb in white/gray to avoid yellow dominance.
    """
    C, H, W = img_CYX.shape

    # Use markers only: ch1, ch2, ch3.
    # If fewer channels exist, fall back safely.
    if C >= 4:
        marker_channels = [1, 2, 3]
    elif C == 3:
        marker_channels = [0, 1, 2]
    elif C == 2:
        marker_channels = [0, 1]
    else:
        marker_channels = [0]

    # Build RGB composite
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Marker palette: green, magenta, cyan-ish/blue
    # Change these later if you want, but NO yellow here.
    colors_rgb = [
        np.array([0.0, 1.0, 0.0], dtype=np.float32),  # green
        np.array([1.0, 0.0, 1.0], dtype=np.float32),  # magenta
        np.array([0.0, 0.7, 1.0], dtype=np.float32),  # cyan/blue
    ]

    for i, ch in enumerate(marker_channels):
        if ch >= C:
            continue
        plane = normalize01(img_CYX[ch])
        color = colors_rgb[i % len(colors_rgb)]
        rgb += plane[..., None] * color[None, None, :]

    rgb = np.clip(rgb, 0, 1)

    # Convert RGB to BGR for OpenCV saving
    out = (rgb * 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # Mask outline — white
    cnts, _ = cv2.findContours(
        (mask_u8 > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)

    # Spokes — light gray, NOT yellow
    angles_sparse = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False)
    for i, a in enumerate(angles_sparse):
        idx = int((i / n_spokes) * len(boundary_r)) % len(boundary_r)
        rmax = float(boundary_r[idx])
        x2 = int(round(cx + rmax * np.cos(a)))
        y2 = int(round(cy + rmax * np.sin(a)))

        cv2.line(
            out,
            (int(round(cx)), int(round(cy))),
            (x2, y2),
            (210, 210, 210),
            1
        )

    # Rings — softer gray dots, NOT magenta/yellow
    dense_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    for frac in ring_fracs:
        for j, a in enumerate(dense_angles):
            idx = int((j / 360) * len(boundary_r)) % len(boundary_r)
            rmax = float(boundary_r[idx])
            r = frac * rmax
            x = int(round(cx + r * np.cos(a)))
            y = int(round(cy + r * np.sin(a)))

            if 0 <= x < W and 0 <= y < H and mask_u8[y, x] > 0:
                cv2.circle(out, (x, y), 1, (180, 180, 180), -1)

    return out

# -------------------------
# Metrics (v0.1)
# -------------------------
def compute_metrics(grid_CRA, r_fracs, edge_band=(0.85,1.0), center_band=(0.0,0.2)):
    """
    Basic quantification that’s genuinely shape-agnostic:
      - radial mean profile (C,R)
      - edge mean, center mean
      - edge/center ratio
      - monotonic gradient score (corr with radius)
      - angular asymmetry at edge (std across angles)
    """
    C, R, A = grid_CRA.shape
    radial_mean = grid_CRA.mean(axis=2)   # C,R

    def band_idx(band):
        lo, hi = band
        idx = np.where((r_fracs >= lo) & (r_fracs <= hi))[0]
        if len(idx) == 0:
            # fallback to nearest
            idx = np.array([np.argmin(np.abs(r_fracs - (lo+hi)/2))])
        return idx

    edge_idx = band_idx(edge_band)
    cen_idx  = band_idx(center_band)

    out = []
    for ch in range(C):
        rm = radial_mean[ch]
        edge_mean = float(rm[edge_idx].mean())
        cen_mean  = float(rm[cen_idx].mean())
        ratio = float(edge_mean / (cen_mean + 1e-6))

        # gradient score: correlation with radius
        r = r_fracs.astype(np.float32)
        corr = float(np.corrcoef(r, rm)[0,1]) if np.std(rm) > 1e-8 else 0.0

        # angular asymmetry at edge: mean std across angles within edge band
        edge_mat = grid_CRA[ch, edge_idx, :]   # (edgeR, A)
        asym = float(edge_mat.mean(axis=0).std())  # std over angles

        out.append({
            "channel": ch,
            "edge_mean": edge_mean,
            "center_mean": cen_mean,
            "edge_center_ratio": ratio,
            "radius_corr": corr,
            "edge_angular_asymmetry": asym
        })
    return pd.DataFrame(out)
#compact topography map summary table helper
def compute_topography_summary(stem, grid, r_fracs):
    """
    Summarize radial topology per channel from the spiderweb grid.
    """
    rows = []
    radial_mean = grid.mean(axis=2)  # (C,R)

    for ch in range(grid.shape[0]):
        rad = radial_mean[ch].astype(np.float32)

        rad_norm, bg, hi = robust_bg_and_scale(rad, bg_q=0.10, hi_q=0.95)
        mass = rad_norm.copy()

        frac_in, frac_mid, frac_out = ring_fracs_from_mass(mass, r_fracs, cuts=(1/3, 2/3))
        r_mean, r_peak, r_width = r_moments_from_mass(mass, r_fracs)

        rows.append({
            "image": stem,
            "channel": ch,
            "frac_inner": frac_in,
            "frac_mid": frac_mid,
            "frac_outer": frac_out,
            "r_mean": r_mean,
            "r_peak": r_peak,
            "r_width": r_width,
            "bg_q10": bg,
            "hi_q95": hi
        })

    return pd.DataFrame(rows)
# -------------------------
# Main
# -------------------------
def process_one(tif_path, out_dir,
                n_angles=240, n_r=200, sigma_r=2.5):
    stem = safe_stem(tif_path)
    df_cells = None
    labels = None

    img = read_ome_as_CYX(tif_path, z_mode=Z_MODE)

    # QC before channel cleaning
    save_raw_channel_qc(img, stem + "_RAW_before_cleaning", out_dir)

    # Remove brightfield if present
    if img.shape[0] == 5:
        img = img[[0, 2, 3, 4], :, :]  # clean ch0=DAPI, ch1=TBXT, ch2=SOX2, ch3=GATA6

    # QC after channel cleaning
    save_raw_channel_qc(img, stem + "_CLEAN_after_BF_removal", out_dir)

    print(f"[CHANNELS] {stem}: cleaned stack shape = {img.shape}")
#in python logic, when there 5 channels its represented as [0, 1, 2, 3, 4] removing ch1 in python c
#corresponds to removing ch2 in human counting, which is the brightfield channel we want to 
#exclude from the analysis and visualizations. This can be sanity checked with the save_raw_channel_qc 
#function which saves the raw channels as a figure for each sample.
    # ---------- MASK ----------
    # Uses your top-of-file config variables:
    # MASK_MODE, DAPI_CHANNEL, THRESH_METHOD, THRESH_PCT
    mask = make_mask(
        img,
        mode=MASK_MODE,                 # "composite" or "dapi"
        dapi_channel=DAPI_CHANNEL,      # int
        thresh=THRESH_METHOD,           # "otsu" or "percentile"
        thresh_pct=THRESH_PCT           # percentile value
    )
    mask_path = os.path.join(out_dir, "debug", f"{stem}_mask_debug.png")

    # Save mask per sample (proper mask output)
    if SAVE_MASKS:
        tiff.imwrite(
            os.path.join(out_dir, "masks", f"{stem}_mask.tif"),
            (mask * 255).astype(np.uint8)
        )

    # ---------- SPIDERWEB ----------
    angles, r_fracs, grid, boundary_r, cx, cy = spiderweb_sample(
        img, mask, n_angles=n_angles, n_r=n_r, sigma_r=sigma_r
    )

   # Rectangular topography (angle vs radius) — best for analysis
    save_topography_maps(stem, out_dir, grid, r_fracs)

    # Polar projection — best for visualizing spatial patterns
    save_polar_topography_maps(stem, out_dir, angles, r_fracs, grid)
    save_composite_rectangular_map(stem, out_dir, grid, r_fracs)
    save_composite_polar_map(stem, out_dir, grid)

    # ---------- DELIVERABLE 2: NUCLEI (cell-to-cell) ----------
    labels = np.zeros(mask.shape, dtype=np.int32)

    if DO_CELLS:
        dapi = img[NUC_DAPI_CHANNEL]

        if NUC_SEGMENTATION_MODE == "hybrid":
            labels = segment_nuclei_hybrid(
                dapi=img[NUC_DAPI_CHANNEL],
                mask=mask
            )

        elif NUC_SEGMENTATION_MODE == "stardist":
            labels = segment_nuclei_stardist(
                dapi=img[NUC_DAPI_CHANNEL],
                mask=mask,
                prob_thresh=STARDIST_PROB_THRESH,
                nms_thresh=STARDIST_NMS_THRESH,
                min_area=STARDIST_MIN_SIZE,
                max_area=STARDIST_MAX_SIZE,
                bbox_pad=STARDIST_BBOX_PAD
            )

        else:
            raise ValueError(
                f"Unknown NUC_SEGMENTATION_MODE: {NUC_SEGMENTATION_MODE}. "
                "Use 'hybrid' or 'stardist'."
            )

        df_cells = extract_cell_table(
            labels, img, stem, cx, cy, angles, boundary_r,
            n_sectors=N_SECTORS,
            ring_edges=RING_EDGES
        )

        # Add explicit channel mapping to the cell CSV
        if df_cells is not None and len(df_cells) > 0:
            df_cells["channel_mapping"] = (
                "clean ch0=DAPI; clean ch1=TBXT; clean ch2=SOX2; clean ch3=GATA6"
            )

            raw_count = int(len(df_cells))
            print(f"[NUCLEI] {stem}: nuclei_count={raw_count}")

            df_cells.to_csv(
                os.path.join(out_dir, "cells", f"{stem}_cells.csv"),
                index=False
            )
        #save instance labels for later cell-resolved maps 
        if np.any(labels > 0):
            tiff.imwrite(
                os.path.join(out_dir, "cells", f"{stem}_labels.tif"),
                labels.astype(np.int32)
            )

        # QC overlays
        if np.any(labels > 0):
            qc_bound = nuclei_overlay_boundaries(dapi, labels, mask)
            cv2.imwrite(
                os.path.join(out_dir, "qc", f"{stem}_nuclei_boundaries.png"),
                qc_bound
            )

            qc_color = nuclei_overlay_colored_instances(dapi, labels, mask, alpha=0.45)
            cv2.imwrite(
                os.path.join(out_dir, "qc", f"{stem}_nuclei_colored.png"),
                qc_color
            )

        # Save CSV only if we actually found nuclei rows
        if df_cells is not None and len(df_cells) > 0:
            df_cells["channel_mapping"] = "clean ch0=DAPI; clean ch1=TBXT; clean ch2=SOX2; clean ch3=GATA6"

            df_cells.to_csv(
                os.path.join(out_dir, "cells", f"{stem}_cells.csv"),
                index=False
            )
# ---------- ORTHOGONAL PROFILES ----------
    theta0 = principal_axis_angle(mask)  # radians

    # major/minor profiles + figure
    save_orthogonal_profiles_figure(stem, out_dir, r_fracs, grid, theta0, n_angles)

    orth_rows = []
    orth_curve_rows = []

    for ch in range(grid.shape[0]):
        pmaj, pmin = orthogonal_profiles_from_grid(
            grid[ch], theta0, n_angles, half_width_bins=3
        )

        # save full curves for plotting later
        for i, r in enumerate(r_fracs):
            orth_curve_rows.append({
                "image": stem,
                "channel": ch,
                "r_frac": float(r),
                "major": float(pmaj[i]),
                "minor": float(pmin[i]),
                "theta0_rad": float(theta0)
            })

        edge_maj = band_mean(pmaj, r_fracs, 0.85, 1.0)
        edge_min = band_mean(pmin, r_fracs, 0.85, 1.0)
        cen_maj  = band_mean(pmaj, r_fracs, 0.0,  0.2)
        cen_min  = band_mean(pmin, r_fracs, 0.0,  0.2)

        orth_rows.append({
            "image": stem,
            "channel": ch,
            "theta0_rad": float(theta0),
            "edge_major": edge_maj,
            "edge_minor": edge_min,
            "center_major": cen_maj,
            "center_minor": cen_min,
            "delta_edge": abs(edge_maj - edge_min),
            "delta_center": abs(cen_maj - cen_min),
            "AI_major_minor": anisotropy_index(pmaj, pmin),
            "corr_major_minor": float(np.corrcoef(pmaj, pmin)[0,1]) if (np.std(pmaj) > 1e-8 and np.std(pmin) > 1e-8) else 0.0,
            "major_peak_r": float(r_fracs[int(np.argmax(pmaj))]),
            "minor_peak_r": float(r_fracs[int(np.argmax(pmin))]),
            "major_total_signal": float(np.sum(pmaj)),
            "minor_total_signal": float(np.sum(pmin)),
            "major_minor_ratio_edge": float(edge_maj / (edge_min + 1e-8)),
            "major_minor_ratio_center": float(cen_maj / (cen_min + 1e-8)),
        })

    # save full orthogonal curves csv
    df_orth_curves = pd.DataFrame(orth_curve_rows)
    df_orth_curves.to_csv(
        os.path.join(out_dir, "profiles", f"{stem}_orthogonal_profiles_curves.csv"),
        index=False
    )

    # summary orthogonal metrics dataframe
    df_orth = pd.DataFrame(orth_rows)

            # ---------- DELIVERABLE A: Pattern fingerprint (per image × channel) ----------
    # Radial mean per channel (averaged over angles)
    radial_mean = grid.mean(axis=2)  # (C,R)

    fp_rows = []
    cuts = (1/3, 2/3)

    for ch in range(grid.shape[0]):
        # 1) Normalize radial curve per channel (robust, channel-agnostic)
        rad = radial_mean[ch].astype(np.float32)

        rad_norm, bg, hi = robust_bg_and_scale(rad, bg_q=0.10, hi_q=0.95)

        # Treat rad_norm as "mass" over radius (nonnegative)
        mass = rad_norm.copy()

        # 2) Ring fractions (inner/mid/outer)
        frac_in, frac_mid, frac_out = ring_fracs_from_mass(mass, r_fracs, cuts=cuts)

        # 3) Compact location + spread
        r_mean, r_peak, r_width = r_moments_from_mass(mass, r_fracs)

        # 4) Orthogonal-aware ring fractions (major/minor)
        pmaj, pmin = orthogonal_profiles_from_grid(grid[ch], theta0, n_angles, half_width_bins=3)
        pmaj_norm, _, _ = robust_bg_and_scale(pmaj.astype(np.float32), bg_q=0.10, hi_q=0.95)
        pmin_norm, _, _ = robust_bg_and_scale(pmin.astype(np.float32), bg_q=0.10, hi_q=0.95)

        maj_in, maj_mid, maj_out = ring_fracs_from_mass(pmaj_norm, r_fracs, cuts=cuts)
        min_in, min_mid, min_out = ring_fracs_from_mass(pmin_norm, r_fracs, cuts=cuts)

        fp_rows.append({
            "image": stem,
            "channel": ch,

            # radial fingerprint
            "frac_inner": frac_in,
            "frac_mid": frac_mid,
            "frac_outer": frac_out,
            "r_mean": r_mean,
            "r_peak": r_peak,
            "r_width": r_width,

            # orthogonal directionality (ring-resolved)
            "frac_inner_major": maj_in,
            "frac_mid_major": maj_mid,
            "frac_outer_major": maj_out,
            "frac_inner_minor": min_in,
            "frac_mid_minor": min_mid,
            "frac_outer_minor": min_out,
            "delta_inner_major_minor": abs(maj_in - min_in),
            "delta_mid_major_minor": abs(maj_mid - min_mid),
            "delta_outer_major_minor": abs(maj_out - min_out),

            # keep your existing anisotropy index too (consistent with df_orth)
            "AI_major_minor": anisotropy_index(pmaj_norm, pmin_norm),

            # debug normalization info (optional but nice)
            "bg_q10": bg,
            "hi_q95": hi,
        })

    df_fp = pd.DataFrame(fp_rows)


    # Save grid backbone
    if SAVE_GRIDS:
        np.save(os.path.join(out_dir, "grids", f"{stem}_grid.npy"), grid)

    # Heatmaps (optional)
    if SAVE_HEATMAPS:
        for ch in range(grid.shape[0]):
            plt.figure(figsize=(8, 4))
            plt.imshow(
                grid[ch],
                aspect="auto",
                origin="lower",
                vmin=0,
                vmax=1,
                cmap=get_channel_cmap(ch)
            )
            plt.colorbar(label="Normalized intensity")
            plt.title(f"{stem} — Spiderweb (r×angle) ch{ch}")
            plt.xlabel("Angle bins")
            plt.ylabel("Radius fraction (0→1)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "heatmaps", f"{stem}_spiderweb_ch{ch}.svg"))
            plt.close()

    # Radial mean profile (mean ± std across angles)
    radial_mean = grid.mean(axis=2)   # (C, R)
    radial_std  = grid.std(axis=2)    # (C, R)

    # -------------------------
    # Save radial profile data
    # -------------------------
    radial_rows = []

    for ch in range(grid.shape[0]):

        m_global = radial_mean[ch].astype(np.float32)
        s_global = radial_std[ch].astype(np.float32)

        m_norm, ch_max = normalize_to_curve_max(m_global)
        s_norm = s_global / (ch_max + 1e-8) if ch_max > 1e-8 else np.zeros_like(s_global)

        for i, r in enumerate(r_fracs):
            radial_rows.append({
                "image": stem,
                "channel": ch,
                "r_frac": float(r),

                # RAW (your current plot)
                "radial_mean_global": float(m_global[i]),
                "radial_std_global": float(s_global[i]),

                # NORMALIZED (Makis version)
                "radial_mean_norm": float(m_norm[i]),
                "radial_std_norm": float(s_norm[i]),

                # metadata
                "channel_max": float(ch_max)
            })

    df_radial = pd.DataFrame(radial_rows)

    df_radial.to_csv(
        os.path.join(out_dir, "profiles", f"{stem}_radial_profiles.csv"),
        index=False
    )
    plt.figure(figsize=(8, 5))
    for ch in range(grid.shape[0]):
        m = radial_mean[ch]
        s = radial_std[ch]
        color = get_channel_color(ch)

        plt.plot(
            r_fracs,
            m,
            label=f"ch{ch}",
            linewidth=2.2,
            color=color
        )

        plt.fill_between(
            r_fracs,
            m - s,
            m + s,
            color=color,
            alpha=0.18
        )

    plt.title(f"{stem} — Radial mean profile (spiderweb)")
    plt.xlabel("Radius fraction (0→1)")
    plt.ylabel("Normalized intensity")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "profiles", f"{stem}_radial_profile.svg"))
    plt.close()

    # NEW ==> Radial mean profile (per-channel max normalized) ==> per-channel max=1 
    plt.figure(figsize=(8, 5))
    for ch in range(grid.shape[0]):
        m_global = radial_mean[ch].astype(np.float32)
        s_global = radial_std[ch].astype(np.float32)

        m_norm, ch_max = normalize_to_curve_max(m_global)
        s_norm = s_global / (ch_max + 1e-8) if ch_max > 1e-8 else np.zeros_like(s_global)

        color = get_channel_color(ch)

        plt.plot(
            r_fracs,
            m_norm,
            label=f"ch{ch}",
            linewidth=2.2,
            color=color
        )
        plt.fill_between(
            r_fracs,
            np.clip(m_norm - s_norm, 0, None),
            m_norm + s_norm,
            color=color,
            alpha=0.2
        )

    plt.title(f"{stem} — Radial mean profile (per-channel max normalized)")
    plt.xlabel("Radius fraction (0→1)")
    plt.ylabel("Intensity (each channel max = 1)")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "profiles", f"{stem}_radial_profile_channelmaxnorm.svg"),
    )
    plt.close()
    # -------------------------
    # Individual channel plots
    # -------------------------
    for ch in range(grid.shape[0]):
        m_global = radial_mean[ch].astype(np.float32)
        s_global = radial_std[ch].astype(np.float32)

        # normalized version (Makis style)
        m_norm, ch_max = normalize_to_curve_max(m_global)
        s_norm = s_global / (ch_max + 1e-8) if ch_max > 1e-8 else np.zeros_like(s_global)

        plt.figure(figsize=(6, 4))

        # global
        #plt.plot(r_fracs, m_global, label="global", linewidth=2)
        #plt.fill_between(r_fracs, m_global - s_global, m_global + s_global, alpha=0.2)

        # normalized max per channel 
        color = get_channel_color(ch)

        plt.plot(
            r_fracs,
            m_norm,
            linestyle="-",
            label="per-channel max",
            linewidth=2.2,
            color=color
        )

        plt.fill_between(
            r_fracs,
            np.clip(m_norm - s_norm, 0, None),
            m_norm + s_norm,
            color=color,
            alpha=0.18
        )

        plt.title(f"{stem} — Channel {ch}")
        plt.xlabel("Radius fraction (0→1)")
        plt.ylabel("Normalized Intensity")
        plt.legend()
        plt.grid(False)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, "profiles", f"{stem}_ch{ch}_radial_profile.svg")
        )
        plt.close()

    # Debug overlay (optional)
    if SAVE_DEBUG:
        dbg = debug_overlay(img, mask, cx, cy, boundary_r)
        debug_path = os.path.join(out_dir, "debug", f"{stem}_spiderweb_overlay.png")
        ok = cv2.imwrite(debug_path, dbg)
    # Metrics table
    dfm = compute_metrics(grid, r_fracs)
    dfm.insert(0, "image", stem)
    df_topo = compute_topography_summary(stem, grid, r_fracs)
    df_orth =  pd.DataFrame(orth_rows)
    return dfm, df_orth, df_fp, df_cells, df_topo, df_radial 

def main():
    ensure_dirs(OUT_DIR)

    tifs = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif*")))
    if not tifs:
        raise SystemExit(f"No TIFFs found in: {INPUT_DIR}")

    all_metrics = []
    all_orth = []
    all_fp = []
    all_cells = []
    all_topo = []
    all_radial_profile_rows = [] #collect radial profiles from every MP 

    for p in tifs:
        try:
            dfm, df_orth, df_fp, df_cells, df_topo, df_radial = process_one(
                p, OUT_DIR,
                n_angles=N_ANGLES,
                n_r=N_R,
                sigma_r=SIGMA_R
            )
            all_metrics.append(dfm)
            all_orth.append(df_orth)
            all_fp.append(df_fp)
            all_topo.append(df_topo)
    
      # NEW: convert this image's long radial table into workbook-friendly rows
            if df_radial is not None and len(df_radial) > 0:
                for ch in sorted(df_radial["channel"].unique()):
                    df_ch = df_radial[df_radial["channel"] == ch].sort_values("r_frac")

                    channel_name = CHANNEL_NAMES.get(int(ch), f"ch{ch}")

                    all_radial_profile_rows.append({
                        "image": df_ch["image"].iloc[0],
                        "channel": int(ch),
                        "channel_name": channel_name,

                        # x-axis
                        "r_fracs": df_ch["r_frac"].to_numpy(dtype=np.float32),

                        # main profile used by the current Excel exporter
                        # This keeps export_radial_profiles_by_channel() working.
                        "profile": df_ch["radial_mean_norm"].to_numpy(dtype=np.float32),

                        # extra useful curves for later analysis
                        "radial_mean_global": df_ch["radial_mean_global"].to_numpy(dtype=np.float32),
                        "radial_std_global": df_ch["radial_std_global"].to_numpy(dtype=np.float32),
                        "radial_mean_norm": df_ch["radial_mean_norm"].to_numpy(dtype=np.float32),
                        "radial_std_norm": df_ch["radial_std_norm"].to_numpy(dtype=np.float32),
                        "channel_max": float(df_ch["channel_max"].iloc[0]),
                    })
                    
            if df_cells is not None and len(df_cells) > 0:
                all_cells.append(df_cells)

            print(f"[OK] {os.path.basename(p)}")
        except Exception as e:
            print(f"[FAIL] {os.path.basename(p)} -> {e}")

    # ----- save base outputs -----
    if all_metrics:
        out_csv = os.path.join(OUT_DIR, "metrics", "metrics.csv")
        pd.concat(all_metrics, ignore_index=True).to_csv(out_csv, index=False)
        print(f"Saved metrics: {out_csv}")

    if all_orth:
        out_csv = os.path.join(OUT_DIR, "metrics", "orthogonal_profiles.csv")
        pd.concat(all_orth, ignore_index=True).to_csv(out_csv, index=False)
        print(f"Saved orthogonal profiles: {out_csv}")

    # ----- save cells (NEW deliverable 2) -----
    df_cells_all = None

    if len(all_cells) > 0:
        df_cells_all = pd.concat(all_cells, ignore_index=True)

        out_cells = os.path.join(OUT_DIR, "metrics", "cells_all.csv")
        df_cells_all.to_csv(out_cells, index=False)
        print(f"Saved cells: {out_cells}")

    if all_topo:
        out_topo = os.path.join(OUT_DIR, "metrics", "topography_summary.csv")
        pd.concat(all_topo, ignore_index=True).to_csv(out_topo, index=False)
        print(f"Saved topography summary: {out_topo}")
    # ----- save radial profiles (NEW) grouped by channel for deliverable 3 -----
    if EXPORT_CHANNEL_PROFILE_WORKBOOK:
        # Makis-style normalized profile: each channel max = 1
        export_radial_profiles_by_channel(
            all_profile_rows=all_radial_profile_rows,
            out_dir=OUT_DIR,
            channel_names=CHANNEL_NAMES,
            error_mode=PROFILE_ERROR_MODE,
            profile_key="radial_mean_norm",
            workbook_label="makis_normalized"
        )

        # Global/non-Makis profile
        export_radial_profiles_by_channel(
            all_profile_rows=all_radial_profile_rows,
            out_dir=OUT_DIR,
            channel_names=CHANNEL_NAMES,
            error_mode=PROFILE_ERROR_MODE,
            profile_key="radial_mean_global",
            workbook_label="global"
        )
        # Grouped radial profile plots: curve-max normalized
        save_grouped_radial_profile_plots(
            all_profile_rows=all_radial_profile_rows,
            out_dir=OUT_DIR,
            channel_names=CHANNEL_NAMES,
            error_mode=PROFILE_ERROR_MODE,
            profile_key="radial_mean_norm",
            profile_label="curve_max_normalized"
        )

        # Grouped radial profile plots: Patterna grid 0–1 normalized
        save_grouped_radial_profile_plots(
            all_profile_rows=all_radial_profile_rows,
            out_dir=OUT_DIR,
            channel_names=CHANNEL_NAMES,
            error_mode=PROFILE_ERROR_MODE,
            profile_key="radial_mean_global",
            profile_label="grid_0to1_normalized"
        )
        if EXPORT_GROUPED_HEATMAPS:
            mean_grid, sd_grid, n_grids = load_and_average_saved_grids(OUT_DIR)

            if mean_grid is not None and n_grids > 0:
                n_r = mean_grid.shape[1]
                n_angles = mean_grid.shape[2]

                r_fracs_grouped = np.linspace(0.0, 1.0, n_r).astype(np.float32)
                angles_grouped = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).astype(np.float32)

                save_grouped_rectangular_heatmaps(
                    mean_grid=mean_grid,
                    out_dir=OUT_DIR,
                    r_fracs=r_fracs_grouped,
                    channel_names=CHANNEL_NAMES,
                    label=f"n{n_grids}"
                )

                save_grouped_polar_heatmaps(
                    mean_grid=mean_grid,
                    out_dir=OUT_DIR,
                    angles=angles_grouped,
                    r_fracs=r_fracs_grouped,
                    channel_names=CHANNEL_NAMES,
                    label=f"n{n_grids}"
                )
        if EXPORT_GROUPED_ORTHOGONAL:
            orthogonal_rows = load_grouped_orthogonal_profiles(
                out_dir=OUT_DIR,
                channel_names=CHANNEL_NAMES,
                half_width_bins=3
            )

            if orthogonal_rows:
                n_imgs_orth = len(set([row["image"] for row in orthogonal_rows]))

                save_grouped_orthogonal_plots(
                    orthogonal_rows=orthogonal_rows,
                    out_dir=OUT_DIR,
                    channel_names=CHANNEL_NAMES,
                    error_mode=PROFILE_ERROR_MODE,
                    label=f"n{n_imgs_orth}"
                )
        # ----- save fingerprints + summary (deliverable A) -----
    if all_fp:
        df_fp_all = pd.concat(all_fp, ignore_index=True)

        out_csv = os.path.join(OUT_DIR, "metrics", "pattern_fingerprints.csv")
        df_fp_all.to_csv(out_csv, index=False)
        print(f"Saved pattern fingerprints: {out_csv}")

        # simplified summary table
        frac_cols = ["frac_inner", "frac_mid", "frac_outer"]
        df_fp_all["dominant_fraction"] = df_fp_all[frac_cols].max(axis=1)

        dom_col = df_fp_all[frac_cols].idxmax(axis=1)
        df_fp_all["dominant_ring"] = dom_col.map({
            "frac_inner": "inner",
            "frac_mid": "mid",
            "frac_outer": "outer"
        })

        keep_cols = [
            "image",
            "channel",
            "dominant_ring",
            "dominant_fraction",
            "frac_inner",
            "frac_mid",
            "frac_outer",
            "r_peak",
            "r_width",
            "AI_major_minor"
        ]
        keep_cols = [c for c in keep_cols if c in df_fp_all.columns]

        df_summary = df_fp_all[keep_cols].copy()

        out_summary = os.path.join(OUT_DIR, "metrics", "fingerprint_summary.csv")
        df_summary.to_csv(out_summary, index=False)
        # ----- save cell phenotype classification -----
    if EXPORT_CELL_PHENOTYPES and df_cells_all is not None and len(df_cells_all) > 0:
        df_classified, df_thresholds = classify_cell_phenotypes(
            df_cells=df_cells_all,
            marker_channels=PHENOTYPE_MARKER_CHANNELS,
            channel_names=CHANNEL_NAMES,
            stat=PHENOTYPE_INTENSITY_STAT,
            threshold_mode=PHENOTYPE_THRESHOLD_MODE,
            percentile=PHENOTYPE_POSITIVE_PERCENTILE,
            percentile_by_channel=PHENOTYPE_POSITIVE_PERCENTILE_BY_CHANNEL,
            manual_thresholds=PHENOTYPE_MANUAL_THRESHOLDS,
            control_images=CONTROL_IMAGES_FOR_THRESHOLDS,
            control_images_by_channel=CONTROL_IMAGES_FOR_THRESHOLDS_BY_CHANNEL,
            control_method=CONTROL_THRESHOLD_METHOD
        )

        summary_by_image, summary_by_ring, summary_by_channel = summarize_cell_phenotypes(
            df_classified=df_classified,
            marker_channels=PHENOTYPE_MARKER_CHANNELS,
            channel_names=CHANNEL_NAMES
        )

        export_cell_phenotype_workbook(
            df_classified=df_classified,
            thresholds=df_thresholds,
            summary_by_image=summary_by_image,
            summary_by_ring=summary_by_ring,
            summary_by_channel=summary_by_channel,
            out_dir=OUT_DIR
        )

        save_cell_phenotype_plots(
            summary_by_image=summary_by_image,
            summary_by_ring=summary_by_ring,
            summary_by_channel=summary_by_channel,
            out_dir=OUT_DIR,
            channel_names=CHANNEL_NAMES
        )

        save_marker_intensity_threshold_diagnostics(
            df_classified=df_classified,
            thresholds=df_thresholds,
            out_dir=OUT_DIR,
            marker_channels=PHENOTYPE_MARKER_CHANNELS,
            channel_names=CHANNEL_NAMES,
            stat=PHENOTYPE_INTENSITY_STAT
        )
        if EXPORT_INTEGRATED_SUMMARY_PANELS:
            save_integrated_spatial_summary_panels(
                df_classified=df_classified,
                out_dir=OUT_DIR,
                channel_names=CHANNEL_NAMES,
                marker_channels=PHENOTYPE_MARKER_CHANNELS
            )
                # --------------------------------------------------
        # Standalone phenotype maps + single-MP marker cell maps
        # --------------------------------------------------
        cell_intensity_display_limits = build_cell_intensity_display_limits(
            df_cells=df_classified,
            marker_channels=PHENOTYPE_MARKER_CHANNELS,
            channel_names=CHANNEL_NAMES,
            stat=CELL_INTENSITY_MAP_STAT,
            mode=CELL_INTENSITY_NORM_MODE,
            pcts=CELL_INTENSITY_NORM_PCTS,
            manual_limits=CELL_INTENSITY_MANUAL_LIMITS,
            out_dir=OUT_DIR
        )

        if EXPORT_STANDALONE_CELL_MAPS:
            for stem, df_img in df_classified.groupby("image"):
                label_path = find_label_path(OUT_DIR, stem)
                mask_path = os.path.join(OUT_DIR, "masks", f"{stem}_mask.tif")

                if label_path is None or not os.path.exists(mask_path):
                    print(f"⚠️ Missing label or mask for standalone cell maps: {stem}")
                    continue

                labels_img = load_instance_labels(label_path)
                mask_img = (tiff.imread(mask_path) > 0).astype(np.uint8)

                save_standalone_cell_phenotype_map(
                    stem=stem,
                    labels=labels_img,
                    mask=mask_img,
                    df_img=df_img,
                    out_dir=OUT_DIR,
                    channel_names=CHANNEL_NAMES,
                    bg=CELL_INTENSITY_MAP_BG
                )

                save_single_mp_marker_cell_maps(
                    stem=stem,
                    labels=labels_img,
                    mask=mask_img,
                    df_img=df_img,
                    out_dir=OUT_DIR,
                    marker_channels=PHENOTYPE_MARKER_CHANNELS,
                    channel_names=CHANNEL_NAMES,
                    stat=CELL_INTENSITY_MAP_STAT,
                    bg=CELL_INTENSITY_MAP_BG,
                    norm_mode=CELL_INTENSITY_NORM_MODE,
                    pcts=CELL_INTENSITY_NORM_PCTS,
                    display_limits_by_channel=cell_intensity_display_limits
                )

        # --------------------------------------------------
        # Grouped disk-rendered marker maps
        # --------------------------------------------------
        if EXPORT_GROUPED_DISK_MARKER_MAPS:
            save_grouped_disk_marker_maps(
                out_dir=OUT_DIR,
                channel_names=CHANNEL_NAMES,
                marker_channels=PHENOTYPE_MARKER_CHANNELS,
                bg=CELL_INTENSITY_MAP_BG
            )
if __name__ == "__main__":
    main()