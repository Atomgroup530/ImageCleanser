# %% [markdown]
# # Image Cleanser Pipeline
# 

# %% [markdown]
# ## Params

# %%
params = {
    "on_Google_drive": False,
    "download_dataset": False,
    "download_model": False,
    "evaluate_model": False,
    "evaluate_dataset": False,
}

# %% [markdown]
# ## R and R_prime

# %% [markdown]
# ### Environment

# %%
%pip install parameters fire lmdb pillow torchvision nltk natsort datasets gdown opencv-python scikit-image numpy

# %%
# Mount to drive
if not params["on_Google_drive"]:
  print ("Skipping mounting Google Drive")
else:
  from google.colab import drive
  drive.mount('/content/drive')
  import os
  # Change to your own Google Drive repo location
  os.chdir('/content/drive/MyDrive/UCSD_COURSES/ECE253/ImageCleanser/repos/ImageCleanser')

# %%
# Download dataset
if not params["download_dataset"]:
  print ("Skipping downloading dataset")
else:
  print("Please manually download zip from https://drive.google.com/file/d/1Gk6TgEfBoSy3j8Cp05AWgCq8qU0zqp9O/view?usp=drive_link")
  print("Then uncomment the below two lines to unzip")
  # !unzip -q datasets.zip -d .
  # print ("Unzip complete")

# %%
if not params["download_model"]:
  print ("Skipping downloading dataset")
else:
  print ("Use the below commands or manually download and upload")
  print ("Put model files to saved_models/")
  # models = {
  #     'None-ResNet-None-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
  #     'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1GGC2IRYEMQviZhqQpbtpeTgHO_IXWetG',
  #     'None-VGG-None-CTC.pth': 'https://drive.google.com/open?id=1FS3aZevvLiGF1PFBm5SkwvVcgI6hJWL9',
  #     'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY',
  #     'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/open?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9',
  #     'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
  # }

  # for k, v in models.items():
  #   doc_id = v[v.find('=')+1:]
  #   !curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$doc_id" > /tmp/intermezzo.html
  #   !curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > $k

  # !ls -al *.pth

# %% [markdown]
# ### Helper Functions

# %%
# Make sure the file structure is:
# -- dataset_name/
# ---- gt.txt
# ---- images/
# ------ 0_Tiredness.png
# ------ ......
# ------ 599_Something.png
def create_lmdb(image_directory_original, gt_file_original, lmdb_output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(lmdb_output_dir, exist_ok=False)


    # Ensure the create_lmdb_dataset.py script exists in your current directory
    if os.path.exists('create_lmdb_dataset.py'):
        print(f"Creating LMDB dataset from {image_directory_original} to {lmdb_output_dir} using {gt_file_original}...")
        # The script create_lmdb_dataset.py expects inputPath, gtFile, and outputPath positionally
        create_lmdb_command = f'python3 create_lmdb_dataset.py {image_directory_original} {gt_file_original} {lmdb_output_dir}'
        !{create_lmdb_command}
        print("LMDB dataset creation finished.")
    else:
        print("Error: create_lmdb_dataset.py not found. Please make sure it's in the current directory.")

# %%
# Scorers
import cv2
import numpy as np
from skimage.morphology import skeletonize

def normalize_score(x, min_val=0, max_val=1):
    """Clamp and normalize to [0,1]."""
    return float(np.clip((x - min_val) / (max_val - min_val + 1e-8), 0, 1))


def calculate_sharp_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad_score = float(np.mean(gx**2 + gy**2))

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    c = 10
    center = mag[h//2 - c:h//2 + c, w//2 - c:w//2 + c]
    high = mag.sum() - center.sum()
    fft_ratio = float(high / (mag.sum() + 1e-8))

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) > 127:
        bw = 255 - bw
    skel = skeletonize(bw // 255).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    if num_labels > 1:
        lengths = stats[1:, cv2.CC_STAT_AREA]
        continuity = float(np.mean(lengths) / (np.std(lengths) + 1e-6))
    else:
        continuity = 0.0

    # normalize sub-scores
    t_norm = np.tanh(np.log1p(tenengrad_score) / 5.0)
    f_norm = normalize_score(fft_ratio, 0, 1)
    c_norm = np.tanh(continuity / 10.0)

    combined = 0.6 * t_norm + 0.3 * f_norm + 0.1 * c_norm
    return normalize_score(combined, 0, 1)


def calculate_low_light_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    meanY = gray.mean()
    dark_ratio = np.mean(gray < 50)
    alpha, beta = 0.7, 0.3
    score = alpha * (1 - meanY / 255.0) + beta * dark_ratio
    return normalize_score(score, 0, 1)


def calculate_contrast_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    p10 = np.percentile(gray, 10)
    p90 = np.percentile(gray, 90)
    contrast = (p90 - p10) / 255.0
    return normalize_score(contrast, 0, 1)


def calculate_noise_score(img, edge_exclude=0.20, ref_sigma=12.0):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    if edge_exclude > 0:
        thr = np.percentile(mag, 100 * (1 - edge_exclude))
        flat_mask = (mag < thr)
    else:
        flat_mask = np.ones_like(gray, dtype=bool)

    g = gray.astype(np.float32)
    flat_mask &= (g > 5) & (g < 250)
    if flat_mask.mean() < 0.10:
        flat_mask = np.ones_like(gray, dtype=bool)

    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    hp_flat = hp[flat_mask]
    mad = np.median(np.abs(hp_flat - np.median(hp_flat)))
    sigma_mad = 1.4826 * mad

    ker = np.array([[1, -2, 1],
                    [-2, 4, -2],
                    [1, -2, 1]], dtype=np.float32)
    L = cv2.filter2D(g, cv2.CV_32F, ker)[1:-1, 1:-1]
    L_use = np.abs(L[flat_mask[1:-1, 1:-1]])
    sigma_immer = (np.sqrt(np.pi / 2.0) / 6.0) * np.mean(L_use) if L_use.size > 0 else sigma_mad

    sigma_est = float(np.median([sigma_mad, sigma_immer]))

    # normalize noise: low noise=0, high noise=1
    norm_noise = sigma_est / (ref_sigma + sigma_est)
    return normalize_score(norm_noise, 0, 1)


# %%
# Computing Scorers
import os
import re
import cv2
import pandas as pd
from tqdm import tqdm

def compute_quality_scores(img_dir, out_csv):
    """
    Compute image quality metrics (sharp, noise, contrast, low-light)
    for all images in a folder and save them as a CSV.

    Args:
        img_dir (str): Path to directory with images.
        out_csv (str): Output CSV path.
    """
    # === Ensure output folder exists ===
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # === Helper: natural numeric sort ===
    def natural_key(name):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

    # === Collect and sort image files ===
    images = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=natural_key
    )
    print(f"📁 Found {len(images)} images in {img_dir}")

    rows = []
    for fname in tqdm(images, desc=f"Analyzing {os.path.basename(img_dir)}"):
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipped unreadable image: {fname}")
            continue

        sharp = calculate_sharp_score(img)
        low_light = calculate_low_light_score(img)
        contrast = calculate_contrast_score(img)
        noise = calculate_noise_score(img)

        # assume filename starts with "idx_label.png"
        try:
            idx = int(os.path.splitext(fname)[0].split("_")[0])
        except ValueError:
            print(f"⚠️ Skipped malformed filename: {fname}")
            continue

        rows.append({
            "idx": idx,
            "filename": fname,
            "sharp_score": sharp,
            "low_light_score": low_light,
            "contrast_score": contrast,
            "noise_score": noise,
        })

    # === Save to CSV ===
    df = pd.DataFrame(rows)
    df = df.sort_values("idx").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved quality scores for {len(df)} images → {out_csv}")

    return df


# %%
import pandas as pd
import os

def combine_results_and_scores(detailed_csv, score_csv, out_dir=None, tag=None):
    """
    Combine recognition results and quality scores CSVs by `idx`,
    save both full and NED<1 filtered versions, and return the merged DataFrames.

    Args:
        detailed_csv (str): Path to detailed results CSV (from test.py).
        score_csv (str): Path to image quality scores CSV.
        out_dir (str): Directory to save combined CSVs (default = same as detailed_csv).
        tag (str): Optional dataset name for output filenames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            (merged_full, merged_filtered)
    """
    # --- Load both CSVs ---
    df_results = pd.read_csv(detailed_csv)
    df_scores = pd.read_csv(score_csv)

    # --- Ensure correct types for merging ---
    df_results["idx"] = df_results["idx"].astype(int)
    df_scores["idx"] = df_scores["idx"].astype(int)

    # --- Merge ---
    merged = pd.merge(df_results, df_scores, on="idx", how="inner")
    print(f"✅ Combined {len(merged)} records from {os.path.basename(detailed_csv)} and {os.path.basename(score_csv)}")

    # --- Prepare output paths ---
    if out_dir is None:
        out_dir = os.path.dirname(detailed_csv)
    os.makedirs(out_dir, exist_ok=True)
    if tag is None:
        tag = os.path.basename(detailed_csv).replace("_detailed_results.csv", "")

    out_full = os.path.join(out_dir, f"{tag}_combined.csv")
    out_filtered = os.path.join(out_dir, f"{tag}_combined_ned_lt1.csv")

    # --- Save results ---
    merged.to_csv(out_full, index=False)
    merged_filtered = merged[merged["norm_edit_distance"] < 1.0].copy()
    merged_filtered.to_csv(out_filtered, index=False)

    print(f"💾 Saved: {out_full}\n💾 Saved filtered (<1.0 NED): {out_filtered}")

    return merged, merged_filtered


# %% [markdown]
# ### Evaluating Model and Datasets
# - M: None-VGG-None-CTC.pth
# - D: MJ600 - 600 pictures out of MJSynth datasets which was used to train M
# - D': D_prime - 400 pictures out of IC13 which is a real-life text dataset, plus 200 pictures taken by phone which are intentionally blurred text pictures

# %%
if not params["evaluate_model"]:
  print ("Skipping evaluating model")
else:
  # Evaluate model on D_prime
  !python3 test.py \
  --eval_data datasets/D_prime/D_prime_lmdb\
  --data_filtering_off \
  --Transformation None --FeatureExtraction VGG --SequenceModeling None --Prediction CTC \
  --saved_model saved_models/None-VGG-None-CTC.pth

# %%
if not params["evaluate_model"]:
  print ("Skipping evaluating model")
else:
  # Evaluate model on MJ600
  !python3 test.py \
  --eval_data datasets/MJ600/MJ600_lmdb \
  --data_filtering_off \
  --Transformation None --FeatureExtraction VGG --SequenceModeling None --Prediction CTC \
  --saved_model saved_models/None-VGG-None-CTC.pth

# %%
if not params["evaluate_dataset"]:
  print ("Skipping evaluating dataset")
else:
  # Compute scorers for the datasets
  compute_quality_scores("./datasets/D_prime/images/","./result/None-VGG-None-CTC.pth/D_prime_quality_scores.csv")

# %%
if not params["evaluate_dataset"]:
  print ("Skipping evaluating dataset")
else:
  compute_quality_scores("./datasets/MJ600/images/","./result/None-VGG-None-CTC.pth/MJ600_quality_scores.csv")

# %%
if not params["evaluate_dataset"]:
  print ("Skipping evaluating dataset")
else:
  # Combine model evaluation and dataset score for better visualization
  combine_results_and_scores("./result/None-VGG-None-CTC.pth/D_prime_detailed_results.csv", "./result/None-VGG-None-CTC.pth/D_prime_quality_scores.csv", "./result/None-VGG-None-CTC.pth/", "D_prime")

# %%
if not params["evaluate_dataset"]:
  print ("Skipping evaluating dataset")
else:
  # Combine model evaluation and dataset score for better visualization
  combine_results_and_scores("./result/None-VGG-None-CTC.pth/MJ600_detailed_results.csv", "./result/None-VGG-None-CTC.pth/MJ600_quality_scores.csv", "./result/None-VGG-None-CTC.pth/", "MJ600")

# %% [markdown]
# ### Visualization
# - Model performance on datasets is evaluated by distribution of confidence and normalized edit distance
# - Image quality in a datset is evaluated by distribution of sharp_score, noise_score, constrast_score, and low_light_score
# - Correlation is shown by mapping model performance metrics to image quality

# %% [markdown]
# #### Model Performance

# %%
import pandas as pd
import matplotlib.pyplot as plt

# --- Load your two result CSVs ---
df1 = pd.read_csv("./result/None-VGG-None-CTC.pth/MJ600_detailed_results.csv")
df2 = pd.read_csv("./result/None-VGG-None-CTC.pth/D_prime_detailed_results.csv")

# Label each dataset
df1["dataset"] = "MJ600"
df2["dataset"] = "D'"

# Combine for summary
df_all = pd.concat([df1, df2], ignore_index=True)

# --- Confidence Distribution ---
plt.figure(figsize=(8, 4))
plt.hist(df1["confidence"], bins=30, alpha=0.5, label="MJ600", density=True)
plt.hist(df2["confidence"], bins=30, alpha=0.5, label="D′", density=True)
plt.title("Distribution of Confidence Scores")
plt.xlabel("Confidence")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# --- Normalized Edit Distance Distribution ---
plt.figure(figsize=(8, 4))
plt.hist(df1["norm_edit_distance"], bins=30, alpha=0.5, label="MJ600", density=True)
plt.hist(df2["norm_edit_distance"], bins=30, alpha=0.5, label="D′", density=True)
plt.title("Distribution of Normalized Edit Distance")
plt.xlabel("Normalized Edit Distance")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# --- Summary statistics ---
summary = df_all.groupby("dataset")[["confidence", "norm_edit_distance"]].describe()
display(summary)


# %% [markdown]
# 

# %%
import pandas as pd
import os

# Load the existing filtered dataframes (NED < 1)
df1 = pd.read_csv("./result/None-VGG-None-CTC.pth/MJ600_combined_ned_lt1.csv")
df2 = pd.read_csv("./result/None-VGG-None-CTC.pth/D_prime_combined_ned_lt1.csv")

# Filter out rows where confidence is 0 or 1
df1_filtered_confidence = df1[(df1['confidence'] > 0) & (df1['confidence'] < 1)].copy()
df2_filtered_confidence = df2[(df2['confidence'] > 0) & (df2['confidence'] < 1)].copy()

# Define output paths
output_dir = "./result/None-VGG-None-CTC.pth/"
os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

output_path_mj600 = os.path.join(output_dir, "MJ600_combined_ned_lt1_conf_gt0lt1.csv")
output_path_dprime = os.path.join(output_dir, "D_prime_combined_ned_lt1_conf_gt0lt1.csv")

# Save the filtered dataframes to new CSV files
df1_filtered_confidence.to_csv(output_path_mj600, index=False)
df2_filtered_confidence.to_csv(output_path_dprime, index=False)

print(f"Saved filtered MJ600 data (NED < 1, Confidence > 0 and < 1) to: {output_path_mj600}")
print(f"Saved filtered D_prime data (NED < 1, Confidence > 0 and < 1) to: {output_path_dprime}")

# %%
import pandas as pd
import matplotlib.pyplot as plt

# --- Load your two result CSVs ---
df1 = pd.read_csv("./result/None-VGG-None-CTC.pth/MJ600_combined_ned_lt1_conf_gt0lt1.csv")
df2 = pd.read_csv("./result/None-VGG-None-CTC.pth/D_prime_combined_ned_lt1_conf_gt0lt1.csv")

# Label each dataset
df1["dataset"] = "MJ600"
df2["dataset"] = "D'"

# Combine for summary
df_all = pd.concat([df1, df2], ignore_index=True)

# --- Confidence Distribution ---
plt.figure(figsize=(8, 4))
plt.hist(df1["confidence"], bins=30, alpha=0.5, label="MJ600", density=True)
plt.hist(df2["confidence"], bins=30, alpha=0.5, label="D′", density=True)
plt.title("Distribution of Confidence Scores")
plt.xlabel("Confidence")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# --- Normalized Edit Distance Distribution ---
plt.figure(figsize=(8, 4))
plt.hist(df1["norm_edit_distance"], bins=30, alpha=0.5, label="MJ600", density=True)
plt.hist(df2["norm_edit_distance"], bins=30, alpha=0.5, label="D′", density=True)
plt.title("Distribution of Normalized Edit Distance")
plt.xlabel("Normalized Edit Distance")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# --- Summary statistics ---
summary = df_all.groupby("dataset")[["confidence", "norm_edit_distance"]].describe()
display(summary)


# %% [markdown]
# #### Image List and Metrics

# %%
# D_prime
from IPython.display import display, HTML
from PIL import Image
import pandas as pd
import io, base64, os

csv_path = "result/None-VGG-None-CTC.pth/D_prime_combined.csv"
img_dir  = "datasets/D_prime/images"

df = pd.read_csv(csv_path)

rows = []
missing = []

for _, row in df.iterrows():
    idx = int(row["idx"])  # filenames start from 1
    gt   = str(row["ground_truth"])
    pred = str(row["prediction"])
    conf = row["confidence"]
    ned  = row["norm_edit_distance"]

    # match by index prefix only
    candidates = [f for f in os.listdir(img_dir) if f.startswith(f"{idx}_")]
    if not candidates:
        missing.append(idx)
        continue
    img_name = candidates[0]
    img_path = os.path.join(img_dir, img_name)

    img = Image.open(img_path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    rows.append({
        "idx": idx,
        "image": f'<img src="data:image/png;base64,{img_b64}" width="160">',
        "filename": img_name,
        "ground_truth": gt,
        "prediction": pred,
        "confidence": f"{conf:.4f}",
        "norm_edit_distance": f"{ned:.4f}",
        "sharp_score":  f"{row['sharp_score']:.3f}",
        "noise_score": f"{row['noise_score']:.3f}",
        "contrast_score": f"{row['contrast_score']:.3f}",
        "low_light_score": f"{row['low_light_score']:.3f}",
    })

display_df = pd.DataFrame(rows)

# --- scrollable HTML output ---
html_table = display_df.to_html(escape=False, index=False)
scrollable_html = f"""
<div style="
    max-height: 600px;
    overflow-y: auto;
    overflow-x: auto;
    border: 1px solid #ccc;
    padding: 8px;
">
{html_table}
</div>
"""
display(HTML(scrollable_html))

if missing:
    print(f"⚠️ Missing {len(missing)} images: {missing[:10]}")


# %%
# MJ600
from IPython.display import display, HTML
from PIL import Image
import pandas as pd
import io, base64, os

csv_path = "result/None-VGG-None-CTC.pth/MJ600_combined.csv"
img_dir  = "datasets/MJ600/images"

df = pd.read_csv(csv_path)

rows = []
missing = []

for _, row in df.iterrows():
    idx = int(row["idx"]) # filenames start from 1
    gt   = str(row["ground_truth"])
    pred = str(row["prediction"])
    conf = row["confidence"]
    ned  = row["norm_edit_distance"]

    # match by index prefix only
    candidates = [f for f in os.listdir(img_dir) if f.startswith(f"{idx}_")]
    if not candidates:
        missing.append(idx)
        continue
    img_name = candidates[0]
    img_path = os.path.join(img_dir, img_name)

    img = Image.open(img_path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    rows.append({
        "idx": idx,
        "image": f'<img src="data:image/png;base64,{img_b64}" width="160">',
        "filename": img_name,
        "ground_truth": gt,
        "prediction": pred,
        "confidence": f"{conf:.4f}",
        "norm_edit_distance": f"{ned:.4f}",
        "sharp_score":  f"{row['sharp_score']:.3f}",
        "noise_score": f"{row['noise_score']:.3f}",
        "contrast_score": f"{row['contrast_score']:.3f}",
        "low_light_score": f"{row['low_light_score']:.3f}",
    })

display_df = pd.DataFrame(rows)

# --- scrollable HTML output ---
html_table = display_df.to_html(escape=False, index=False)
scrollable_html = f"""
<div style="
    max-height: 600px;
    overflow-y: auto;
    overflow-x: auto;
    border: 1px solid #ccc;
    padding: 8px;
">
{html_table}
</div>
"""
display(HTML(scrollable_html))

if missing:
    print(f"⚠️ Missing {len(missing)} images: {missing[:10]}")


# %% [markdown]
# #### Image List and Metrics for NED < 1

# %%
# D_prime
from IPython.display import display, HTML
from PIL import Image
import pandas as pd
import io, base64, os

csv_path = "result/None-VGG-None-CTC.pth/D_prime_combined_ned_lt1.csv"
img_dir  = "datasets/D_prime/images"

df = pd.read_csv(csv_path)

rows = []
missing = []

for _, row in df.iterrows():
    idx = int(row["idx"])  # filenames start from 1
    gt   = str(row["ground_truth"])
    pred = str(row["prediction"])
    conf = row["confidence"]
    ned  = row["norm_edit_distance"]

    # match by index prefix only
    candidates = [f for f in os.listdir(img_dir) if f.startswith(f"{idx}_")]
    if not candidates:
        missing.append(idx)
        continue
    img_name = candidates[0]
    img_path = os.path.join(img_dir, img_name)

    img = Image.open(img_path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    rows.append({
        "idx": idx,
        "image": f'<img src="data:image/png;base64,{img_b64}" width="160">',
        "filename": img_name,
        "ground_truth": gt,
        "prediction": pred,
        "confidence": f"{conf:.4f}",
        "norm_edit_distance": f"{ned:.4f}",
        "sharp_score":  f"{row['sharp_score']:.3f}",
        "noise_score": f"{row['noise_score']:.3f}",
        "contrast_score": f"{row['contrast_score']:.3f}",
        "low_light_score": f"{row['low_light_score']:.3f}",
    })

display_df = pd.DataFrame(rows)

# --- scrollable HTML output ---
html_table = display_df.to_html(escape=False, index=False)
scrollable_html = f"""
<div style="
    max-height: 600px;
    overflow-y: auto;
    overflow-x: auto;
    border: 1px solid #ccc;
    padding: 8px;
">
{html_table}
</div>
"""
display(HTML(scrollable_html))

if missing:
    print(f"⚠️ Missing {len(missing)} images: {missing[:10]}")


# %%
# D_prime
from IPython.display import display, HTML
from PIL import Image
import pandas as pd
import io, base64, os

csv_path = "result/None-VGG-None-CTC.pth/MJ600_combined_ned_lt1.csv"
img_dir  = "datasets/MJ600/images"

df = pd.read_csv(csv_path)

rows = []
missing = []

for _, row in df.iterrows():
    idx = int(row["idx"])  # filenames start from 1
    gt   = str(row["ground_truth"])
    pred = str(row["prediction"])
    conf = row["confidence"]
    ned  = row["norm_edit_distance"]

    # match by index prefix only
    candidates = [f for f in os.listdir(img_dir) if f.startswith(f"{idx}_")]
    if not candidates:
        missing.append(idx)
        continue
    img_name = candidates[0]
    img_path = os.path.join(img_dir, img_name)

    img = Image.open(img_path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    rows.append({
        "idx": idx,
        "image": f'<img src="data:image/png;base64,{img_b64}" width="160">',
        "filename": img_name,
        "ground_truth": gt,
        "prediction": pred,
        "confidence": f"{conf:.4f}",
        "norm_edit_distance": f"{ned:.4f}",
        "sharp_score":  f"{row['sharp_score']:.3f}",
        "noise_score": f"{row['noise_score']:.3f}",
        "contrast_score": f"{row['contrast_score']:.3f}",
        "low_light_score": f"{row['low_light_score']:.3f}",
    })

display_df = pd.DataFrame(rows)

# --- scrollable HTML output ---
html_table = display_df.to_html(escape=False, index=False)
scrollable_html = f"""
<div style="
    max-height: 600px;
    overflow-y: auto;
    overflow-x: auto;
    border: 1px solid #ccc;
    padding: 8px;
">
{html_table}
</div>
"""
display(HTML(scrollable_html))

if missing:
    print(f"⚠️ Missing {len(missing)} images: {missing[:10]}")


# %% [markdown]
# #### Image Scorer Distribution

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- paths to your combined CSVs ---
csv_dprime = "result/None-VGG-None-CTC.pth/D_prime_combined.csv"
csv_mj600  = "result/None-VGG-None-CTC.pth/MJ600_combined.csv"

# --- load ---
df_dprime = pd.read_csv(csv_dprime)
df_mj600  = pd.read_csv(csv_mj600)

# --- label datasets ---
df_dprime["dataset"] = "D_prime"
df_mj600["dataset"]  = "MJ600"

# --- concatenate for plotting ---
df_all = pd.concat([df_dprime, df_mj600], ignore_index=True)

# --- columns to compare ---
metrics = ["sharp_score", "noise_score", "contrast_score", "low_light_score"]
titles  = ["Sharp", "Noise", "Contrast", "Low-light"]

# --- plot ---
sns.set(style="whitegrid")

for m, t in zip(metrics, titles):
    plt.figure(figsize=(8, 4))
    sns.kdeplot(
        data=df_all, x=m, hue="dataset", fill=True, common_norm=False,
        alpha=0.5, linewidth=1.2
    )
    plt.title(f"{t} Score Distribution: D_prime vs MJ600")
    plt.xlabel("Normalized score (0–1)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- paths to your combined CSVs ---
csv_dprime = "result/None-VGG-None-CTC.pth/D_prime_combined.csv"
csv_mj600  = "result/None-VGG-None-CTC.pth/MJ600_combined.csv"

# --- load ---
df_dprime = pd.read_csv(csv_dprime)
df_mj600  = pd.read_csv(csv_mj600)

# --- label datasets ---
df_dprime["dataset"] = "D_prime"
df_mj600["dataset"]  = "MJ600"

# --- concatenate for plotting ---
df_all = pd.concat([df_dprime, df_mj600], ignore_index=True)

# --- columns to compare ---
performance_metrics = ["confidence", "norm_edit_distance"]
quality_metrics = ["sharp_score", "noise_score", "contrast_score", "low_light_score"]
quality_titles  = ["Sharpness", "Noise", "Contrast", "Low-light"]

# --- plot ---
sns.set(style="whitegrid")

for perf_metric in performance_metrics:
    for quality_metric, quality_title in zip(quality_metrics, quality_titles):
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=df_dprime,
            x=quality_metric,
            y=perf_metric,
            scatter_kws={'alpha':0.6, 's':15},
            label="D_prime",
            ci=95  # Confidence interval
        )
        sns.regplot(
            data=df_mj600,
            x=quality_metric,
            y=perf_metric,
            scatter_kws={'alpha':0.6, 's':15},
            label="MJ600",
            ci=95 # Confidence interval
        )
        plt.title(f"Correlation: {perf_metric.replace('_', ' ').title()} vs {quality_title}")
        plt.xlabel(f"{quality_title} Score (0-1)")
        plt.ylabel(perf_metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        plt.show()

# %% [markdown]
# ## Contrast New

# %%
# Clova dataset / collate utilities
from dataset import RawDataset, AlignCollate
from utils import CTCLabelConverter
from model import Model
from nltk.metrics.distance import edit_distance


# %%
from types import SimpleNamespace

opt = SimpleNamespace()

# MODEL
opt.saved_model = "saved_models/None-VGG-None-CTC.pth"
opt.batch_max_length = 25
opt.imgH = 32
opt.imgW = 100
opt.rgb = False
opt.character = "0123456789abcdefghijklmnopqrstuvwxyz"
opt.sensitive = False
opt.PAD = True
opt.workers = 0

# ARCHITECTURE
opt.Transformation = "None"
opt.FeatureExtraction = "VGG"
opt.SequenceModeling = "None"
opt.Prediction = "CTC"

# CHANNELS
opt.num_fiducial = 20
opt.input_channel = 1
opt.output_channel = 512
opt.hidden_size = 256


# %%
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ocr_model(opt):
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt).to(device)
    model = torch.nn.DataParallel(model)

    print("Loading model:", opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device, weights_only=False))
    model.eval()

    return model, converter


# %%
import numpy as np
import cv2

def _contrast_pr90_pr10(gray):
    p10 = np.percentile(gray, 10)
    p90 = np.percentile(gray, 90)
    return float((p90 - p10) / 255.0)

def compute_contrast_metrics(gray_enh, gray_orig=None):
    gray_enh = gray_enh.astype(np.uint8)

    # 1) sigma (std)
    sigma = float(np.std(gray_enh))

    # 2) SSIM (fallback to 0 if no orig)
    ssim_val = 0.0
    if gray_orig is not None:
        try:
            from skimage.metrics import structural_similarity as ssim
            gray_orig = gray_orig.astype(np.uint8)
            ssim_val = float(ssim(gray_orig, gray_enh, data_range=255))
        except Exception:
            ssim_val = 0.0

    # 3) Tenengrad (your original: sum of Sobel energy)
    gx = cv2.Sobel(gray_enh, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_enh, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.sum(gx**2 + gy**2))

    # 4) delta_mu brightness shift
    if gray_orig is None:
        delta_mu = 0.0
    else:
        gray_orig = gray_orig.astype(np.uint8)
        delta_mu = float(abs(float(np.mean(gray_enh)) - float(np.mean(gray_orig))))

    # 5) proposal contrast score
    contrast_pr = _contrast_pr90_pr10(gray_enh)
    contrast_gain = 0.0 if gray_orig is None else float(contrast_pr - _contrast_pr90_pr10(gray_orig))

    return {
        "sigma": sigma,
        "ssim": ssim_val,
        "tenengrad": tenengrad,
        "delta_mu": delta_mu,
        "contrast_pr": contrast_pr,
        "contrast_gain": contrast_gain,
    }


# %%
def enhance_he(gray):
    return cv2.equalizeHist(gray.astype(np.uint8))

def enhance_clahe(gray, clip=2.0, grid=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(grid))
    return clahe.apply(gray.astype(np.uint8))

# --- RSWHE (practical RSWHE-like) ---
def _segment_weighted_he_lut(lo, hi, pdf, alpha=0.6):
    levels = np.arange(lo, hi + 1)
    p = pdf[levels].astype(np.float64)

    w = np.power(p + 1e-12, alpha)
    w = w / (w.sum() + 1e-12)
    cdf = np.cumsum(w)

    mapping = np.round(lo + (hi - lo) * cdf).astype(np.uint8)

    lut = np.arange(256, dtype=np.uint8)
    lut[levels] = mapping
    return lut

def enhance_rswhe(gray, recursion=2, alpha=0.6):
    g = gray.astype(np.uint8)

    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    pdf = hist / (hist.sum() + 1e-12)

    segments = [(0, 255)]
    for _ in range(int(recursion)):
        new_segments = []
        for lo, hi in segments:
            if lo >= hi:
                new_segments.append((lo, hi))
                continue
            levels = np.arange(lo, hi + 1)
            p = pdf[levels]
            mu = int(np.round((levels * p).sum() / (p.sum() + 1e-12)))
            mu = int(np.clip(mu, lo, hi))

            if mu <= lo or mu >= hi:
                new_segments.append((lo, hi))
            else:
                new_segments.append((lo, mu))
                new_segments.append((mu + 1, hi))
        segments = new_segments

    lut = np.arange(256, dtype=np.uint8)
    for lo, hi in segments:
        seg_lut = _segment_weighted_he_lut(lo, hi, pdf, alpha=float(alpha))
        lut[lo:hi+1] = seg_lut[lo:hi+1]

    return cv2.LUT(g, lut)


# %%
def refine_text_mask(gray, method="otsu", k=2, erode_iter=1, open_iter=1):
    g = gray.astype(np.uint8)

    if method == "otsu":
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bw = cv2.adaptiveThreshold(g, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 5)

    # assume darker text: make text pixels be 1
    if bw.mean() > 127:
        bw = 255 - bw

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    if erode_iter > 0:
        bw = cv2.erode(bw, ker, iterations=int(erode_iter))
    if open_iter > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker, iterations=int(open_iter))

    return (bw > 0).astype(np.float32)

def blend_enhancement(gray_orig, gray_enh, mask, strength=0.85):
    g0 = gray_orig.astype(np.float32)
    ge = gray_enh.astype(np.float32)
    m = np.clip(mask.astype(np.float32), 0, 1)
    out = g0 * (1 - strength*m) + ge * (strength*m)
    return np.clip(out, 0, 255).astype(np.uint8)

def enhance_clahe_masked(gray, clip=2.0, grid=(8,8), mask_k=2, strength=0.85):
    base = gray.astype(np.uint8)
    enh = enhance_clahe(base, clip=clip, grid=grid)
    mask = refine_text_mask(base, method="otsu", k=mask_k, erode_iter=1, open_iter=1)
    return blend_enhancement(base, enh, mask, strength=strength)

def enhance_rswhe_masked(gray, recursion=2, alpha=0.6, mask_k=2, strength=0.85):
    base = gray.astype(np.uint8)
    enh = enhance_rswhe(base, recursion=recursion, alpha=alpha)
    mask = refine_text_mask(base, method="otsu", k=mask_k, erode_iter=1, open_iter=1)
    return blend_enhancement(base, enh, mask, strength=strength)


# %%
from pathlib import Path

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def rebuild_enhanced_dataset(src_root, out_root, enhancer_fn):
    src_root = Path(src_root)
    out_root = Path(out_root)

    src_img = src_root / "images"
    src_gt  = src_root / "gt.txt"

    _ensure_dir(out_root / "images")
    (out_root / "gt.txt").write_text(src_gt.read_text(encoding="utf-8"), encoding="utf-8")

    img_paths = sorted([p for p in src_img.rglob("*") if p.suffix.lower() in [".png",".jpg",".jpeg",".bmp"]])
    for p in img_paths:
        rel = p.relative_to(src_img)
        out_p = out_root / "images" / rel
        _ensure_dir(out_p.parent)

        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        out_g = enhancer_fn(g)
        cv2.imwrite(str(out_p), out_g)

    print("[OK] Built:", out_root)
    return out_root


# %%
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

def run_scoring_and_eval_on_folder(method_name, dataset_root, opt, baseline_root=None):
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images"
    gt_path = dataset_root / "gt.txt"

    baseline_img_dir = None
    if baseline_root is not None:
        baseline_img_dir = Path(baseline_root) / "images"

    # Read GT
    label_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            rel, label = line.strip().split("\t")
            label_map[rel] = label

    # Data loader (same as yours)
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    raw_dataset = RawDataset(root=str(img_dir), opt=opt)
    loader = DataLoader(raw_dataset, batch_size=192, shuffle=False, num_workers=0, collate_fn=align_collate)

    # Model (same as yours)
    model, converter = load_ocr_model(opt)

    rows = []

    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc=f"[{method_name}] Eval"):
            imgs = imgs.to(device)
            bsz = imgs.size(0)

            txt_for_pred = torch.zeros(bsz, opt.batch_max_length + 1, dtype=torch.long).to(device)

            preds = model(imgs, txt_for_pred)
            preds_prob = torch.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            _, preds_index = preds.max(2)

            preds_str = converter.decode(preds_index, torch.IntTensor([preds.size(1)] * bsz))

            for img_path, pred, prob_seq in zip(paths, preds_str, preds_max_prob):
                rel = os.path.relpath(img_path, start=str(img_dir)).replace("\\", "/")
                gt = label_map[rel]

                conf = float(prob_seq.cumprod(dim=0)[-1])

                if len(gt) == 0 or len(pred) == 0:
                    ned = 0.0
                elif len(gt) >= len(pred):
                    ned = 1 - edit_distance(pred, gt) / len(gt)
                else:
                    ned = 1 - edit_distance(pred, gt) / len(pred)

                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                gray0 = None
                if baseline_img_dir is not None:
                    base_path = baseline_img_dir / rel
                    if base_path.exists():
                        gray0 = cv2.imread(str(base_path), cv2.IMREAD_GRAYSCALE)

                dip = compute_contrast_metrics(gray, gray0)
                is_correct = int(pred == gt)

                rows.append({
                    "method": method_name,
                    "image_rel": rel,
                    "gt": gt,
                    "pred": pred,
                    "confidence": conf,
                    "ned": float(ned),
                    "accuracy": is_correct,

                    # keep your old columns
                    "sigma": dip["sigma"],
                    "ssim": dip["ssim"],
                    "tenengrad": dip["tenengrad"],
                    "delta_mu": dip["delta_mu"],

                    # add new columns (proposal-aligned)
                    "contrast_pr": dip["contrast_pr"],
                    "contrast_gain": dip["contrast_gain"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(dataset_root / f"results_{method_name}.csv", index=False)
    print(f"Saved: {dataset_root}/results_{method_name}.csv")
    return df


# %%
from pathlib import Path

src_root = Path("datasets/D_prime")
baseline_root = src_root

# 1) Build enhanced datasets
he_root = Path("datasets/D_prime_HE")
clahe_root = Path("datasets/D_prime_CLAHE")
rswhe_root = Path("datasets/D_prime_RSWHE")
clahe_mask_root = Path("datasets/D_prime_CLAHE_MASK")
rswhe_mask_root = Path("datasets/D_prime_RSWHE_MASK")

# If you already built HE/CLAHE folders earlier, you can skip rebuild for those.
rebuild_enhanced_dataset(src_root, he_root, enhance_he)
rebuild_enhanced_dataset(src_root, clahe_root, lambda g: enhance_clahe(g, clip=2.0, grid=(8,8)))
rebuild_enhanced_dataset(src_root, rswhe_root, lambda g: enhance_rswhe(g, recursion=2, alpha=0.6))

# educated erosion experiment (mask + erosion/open)
rebuild_enhanced_dataset(src_root, clahe_mask_root, lambda g: enhance_clahe_masked(g, clip=2.0, grid=(8,8), mask_k=2, strength=0.85))
rebuild_enhanced_dataset(src_root, rswhe_mask_root, lambda g: enhance_rswhe_masked(g, recursion=2, alpha=0.6, mask_k=2, strength=0.85))

# 2) Evaluate (compare metrics vs baseline where meaningful)
print("Evaluating baseline...")
df_base = run_scoring_and_eval_on_folder("baseline", baseline_root, opt, baseline_root=None)

print("Evaluating HE...")
df_he = run_scoring_and_eval_on_folder("HE", he_root, opt, baseline_root=baseline_root)

print("Evaluating CLAHE...")
df_clahe = run_scoring_and_eval_on_folder("CLAHE", clahe_root, opt, baseline_root=baseline_root)

print("Evaluating RSWHE...")
df_rswhe = run_scoring_and_eval_on_folder("RSWHE", rswhe_root, opt, baseline_root=baseline_root)

print("Evaluating CLAHE_MASK...")
df_clahe_m = run_scoring_and_eval_on_folder("CLAHE_MASK", clahe_mask_root, opt, baseline_root=baseline_root)

print("Evaluating RSWHE_MASK...")
df_rswhe_m = run_scoring_and_eval_on_folder("RSWHE_MASK", rswhe_mask_root, opt, baseline_root=baseline_root)

# quick summary
def summarize(df):
    return pd.Series({
        "mean_accuracy": df["accuracy"].mean(),
        "mean_ned": df["ned"].mean(),
        "mean_conf": df["confidence"].mean(),
        "mean_contrast_pr": df["contrast_pr"].mean(),
        "mean_contrast_gain": df["contrast_gain"].mean(),
        "mean_delta_mu": df["delta_mu"].mean(),
    })


summary = pd.DataFrame({
    "baseline": summarize(df_base),
    "HE": summarize(df_he),
    "CLAHE": summarize(df_clahe),
    "RSWHE": summarize(df_rswhe),
    "CLAHE_MASK": summarize(df_clahe_m),
    "RSWHE_MASK": summarize(df_rswhe_m),
}).T

print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
summary.to_csv("contrast_summary.csv", index=True)


# %%
import pandas as pd
from pathlib import Path

def load_results_csv(dataset_dir, method_name):
    p = Path(dataset_dir) / f"results_{method_name}.csv"
    assert p.exists(), f"Missing {p}"
    return pd.read_csv(p)

# point these to your actual dataset folders
base_dir = "datasets/D_prime"
he_dir = "datasets/D_prime_HE"
clahe_dir = "datasets/D_prime_CLAHE"
rswhe_dir = "datasets/D_prime_RSWHE"
clahe_m_dir = "datasets/D_prime_CLAHE_MASK"
rswhe_m_dir = "datasets/D_prime_RSWHE_MASK"

df_base  = load_results_csv(base_dir, "baseline")
df_he    = load_results_csv(he_dir, "HE")
df_clahe = load_results_csv(clahe_dir, "CLAHE")
df_rswhe = load_results_csv(rswhe_dir, "RSWHE")
df_cm    = load_results_csv(clahe_m_dir, "CLAHE_MASK")
df_rm    = load_results_csv(rswhe_m_dir, "RSWHE_MASK")

df_all = pd.concat([df_base, df_he, df_clahe, df_rswhe, df_cm, df_rm], ignore_index=True)
df_all.head()


# %%
def pick_story_images(df_base, df_he, df_rm, k=3):
    # Merge BASE + HE
    m = df_base[["image_rel","ned","confidence"]].merge(
        df_he[["image_rel","ned","confidence","delta_mu","contrast_gain"]],
        on="image_rel", suffixes=("_base","_he")
    )

    # After merge:
    # - base: ned_base, confidence_base
    # - HE:   ned_he, confidence_he
    # - HE-only cols likely still named: delta_mu, contrast_gain
    # Rename them to delta_mu_he / contrast_gain_he explicitly:
    m = m.rename(columns={
        "delta_mu": "delta_mu_he",
        "contrast_gain": "contrast_gain_he",
    })

    # Merge in RSWHE_MASK and then rename its columns to *_rm
    m = m.merge(
        df_rm[["image_rel","ned","confidence","delta_mu","contrast_gain"]],
        on="image_rel", suffixes=("", "_rm")
    ).rename(columns={
        "ned": "ned_rm",
        "confidence": "confidence_rm",
        "delta_mu": "delta_mu_rm",
        "contrast_gain": "contrast_gain_rm",
    })

    # Score columns
    m["ned_drop_he"] = m["ned_base"] - m["ned_he"]
    m["ned_gain_rm_vs_he"] = m["ned_rm"] - m["ned_he"]
    m["conf_drop_he"] = m["confidence_base"] - m["confidence_he"]

    picks = {}
    picks["he_most_damage"] = m.sort_values("ned_drop_he", ascending=False).head(k)["image_rel"].tolist()
    picks["mask_best_recovery"] = m.sort_values("ned_gain_rm_vs_he", ascending=False).head(k)["image_rel"].tolist()
    picks["he_max_delta_mu"] = m.sort_values("delta_mu_he", ascending=False).head(k)["image_rel"].tolist()
    picks["he_max_contrast_gain"] = m.sort_values("contrast_gain_he", ascending=False).head(k)["image_rel"].tolist()

    return picks, m

picks, merged = pick_story_images(df_base, df_he, df_rm, k=3)
picks


# %%
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

def load_gray(dataset_dir, image_rel):
    p = Path(dataset_dir) / "images" / image_rel
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert g is not None, f"Cannot read {p}"
    return g

def save_erosion_demo(image_rel, out_name=None):
    gray = load_gray(base_dir, image_rel)

    rswhe = enhance_rswhe(gray, recursion=2, alpha=0.6)
    mask = refine_text_mask(gray, k=2)
    rswhe_mask = enhance_rswhe_masked(gray, recursion=2, alpha=0.6, mask_k=2, strength=0.85)

    diff = np.abs(rswhe.astype(np.int32) - rswhe_mask.astype(np.int32))

    fig, axs = plt.subplots(1, 5, figsize=(18,4))
    axs[0].imshow(gray, cmap='gray'); axs[0].set_title("Original")
    axs[1].imshow(rswhe, cmap='gray'); axs[1].set_title("RSWHE")
    axs[2].imshow(mask, cmap='gray'); axs[2].set_title("Text Mask (Erode/Open)")
    axs[3].imshow(rswhe_mask, cmap='gray'); axs[3].set_title("RSWHE_MASK")
    axs[4].imshow(diff, cmap='hot'); axs[4].set_title("|RSWHE − Masked|")

    for ax in axs: ax.axis("off")
    plt.tight_layout()

    if out_name is None:
        out_name = image_rel.replace("/", "_").replace("\\", "_")
    out_path = fig_dir / f"erosion_demo_{out_name}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

# Save 3 strongest "mask recovery" examples
for rel in picks["mask_best_recovery"]:
    p = save_erosion_demo(rel)
    print("Saved:", p)


# %%
import matplotlib.pyplot as plt

def save_scatter_delta_mu_vs_ned(summary_df):
    plt.figure(figsize=(6,5))
    plt.scatter(summary_df["mean_delta_mu"], summary_df["mean_ned"], s=120)

    for _, r in summary_df.iterrows():
        plt.text(r["mean_delta_mu"]+0.5, r["mean_ned"], r["method"], fontsize=9)

    plt.xlabel("Mean Brightness Drift Δμ")
    plt.ylabel("Mean NED")
    plt.title("OCR Accuracy vs Brightness Drift")
    plt.grid(True)
    out = fig_dir / "scatter_delta_mu_vs_ned.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out

def save_bar_contrast_gain(summary_df):
    plt.figure(figsize=(8,4))
    plt.bar(summary_df["method"], summary_df["mean_contrast_gain"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean Contrast Gain (PR90−PR10)")
    plt.title("Contrast Gain by Method")
    out = fig_dir / "bar_contrast_gain.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out

def save_bar_ned(summary_df):
    plt.figure(figsize=(8,4))
    plt.bar(summary_df["method"], summary_df["mean_ned"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean NED")
    plt.title("OCR Accuracy (NED) by Method")
    out = fig_dir / "bar_ned.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out

def save_bar_accuracy(summary_df):
    plt.figure(figsize=(8,4))
    plt.bar(summary_df["method"], summary_df["mean_accuracy"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Accuracy (Exact Match)")
    plt.title("OCR Accuracy by Method")
    out = fig_dir / "bar_accuracy.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out

def save_confidence_boxplot(df_all):
    # If seaborn isn't allowed/installed, we'll do a simple matplotlib boxplot per method
    methods = df_all["method"].unique().tolist()
    data = [df_all[df_all["method"]==m]["confidence"].values for m in methods]

    plt.figure(figsize=(10,4))
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Confidence")
    plt.title("OCR Confidence Distribution by Method")
    out = fig_dir / "box_confidence.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out

# Build summary_df from your printed table (or recompute from df objects)
summary_df = pd.DataFrame([
    {"method":"baseline",
     "mean_accuracy": df_base["accuracy"].mean(),
     "mean_ned": df_base["ned"].mean(),
     "mean_conf": df_base["confidence"].mean(),
     "mean_contrast_pr": df_base["contrast_pr"].mean(),
     "mean_contrast_gain": df_base["contrast_gain"].mean(),
     "mean_delta_mu": df_base["delta_mu"].mean()},
    
    {"method":"HE",
     "mean_accuracy": df_he["accuracy"].mean(),
     "mean_ned": df_he["ned"].mean(),
     "mean_conf": df_he["confidence"].mean(),
     "mean_contrast_pr": df_he["contrast_pr"].mean(),
     "mean_contrast_gain": df_he["contrast_gain"].mean(),
     "mean_delta_mu": df_he["delta_mu"].mean()},
    
    {"method":"CLAHE",
     "mean_accuracy": df_clahe["accuracy"].mean(),
     "mean_ned": df_clahe["ned"].mean(),
     "mean_conf": df_clahe["confidence"].mean(),
     "mean_contrast_pr": df_clahe["contrast_pr"].mean(),
     "mean_contrast_gain": df_clahe["contrast_gain"].mean(),
     "mean_delta_mu": df_clahe["delta_mu"].mean()},
    
    {"method":"RSWHE",
     "mean_accuracy": df_rswhe["accuracy"].mean(),
     "mean_ned": df_rswhe["ned"].mean(),
     "mean_conf": df_rswhe["confidence"].mean(),
     "mean_contrast_pr": df_rswhe["contrast_pr"].mean(),
     "mean_contrast_gain": df_rswhe["contrast_gain"].mean(),
     "mean_delta_mu": df_rswhe["delta_mu"].mean()},
    
    {"method":"CLAHE_MASK",
     "mean_accuracy": df_cm["accuracy"].mean(),
     "mean_ned": df_cm["ned"].mean(),
     "mean_conf": df_cm["confidence"].mean(),
     "mean_contrast_pr": df_cm["contrast_pr"].mean(),
     "mean_contrast_gain": df_cm["contrast_gain"].mean(),
     "mean_delta_mu": df_cm["delta_mu"].mean()},
    
    {"method":"RSWHE_MASK",
     "mean_accuracy": df_rm["accuracy"].mean(),
     "mean_ned": df_rm["ned"].mean(),
     "mean_conf": df_rm["confidence"].mean(),
     "mean_contrast_pr": df_rm["contrast_pr"].mean(),
     "mean_contrast_gain": df_rm["contrast_gain"].mean(),
     "mean_delta_mu": df_rm["delta_mu"].mean()},
])


print("Saved:", save_scatter_delta_mu_vs_ned(summary_df))
print("Saved:", save_bar_contrast_gain(summary_df))
print("Saved:", save_bar_ned(summary_df))
print("Saved:", save_confidence_boxplot(df_all))
print("Saved:", save_bar_accuracy(summary_df))


# %%
# === Single image: all enhancement methods side by side ===
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

image_rel = "402_3.png"   # change to any filename in images/

def load_gray_flat(image_rel, base_dir=base_dir):
    p = Path(base_dir) / "images" / image_rel
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert g is not None, f"Cannot read {p}"
    return g

gray = load_gray_flat(image_rel)

methods = [
    ("Original", lambda g: g),
    ("HE", enhance_he),
    ("CLAHE", lambda g: enhance_clahe(g, clip=2.0, grid=(8,8))),
    ("RSWHE", lambda g: enhance_rswhe(g, recursion=2, alpha=0.6)),
    ("CLAHE_MASK", lambda g: enhance_clahe_masked(g, clip=2.0, grid=(8,8), mask_k=2, strength=0.85)),
    ("RSWHE_MASK", lambda g: enhance_rswhe_masked(g, recursion=2, alpha=0.6, mask_k=2, strength=0.85)),
]

fig, axes = plt.subplots(1, len(methods), figsize=(3.2*len(methods), 3))

for ax, (name, fn) in zip(axes, methods):
    out = fn(gray)
    ax.imshow(out, cmap="gray", vmin=0, vmax=255)
    ax.set_title(name)
    ax.axis("off")

plt.suptitle(f"Contrast Enhancement Comparison: {image_rel}", y=1.05)
plt.tight_layout()
plt.show()


# %%
# === Single image: all enhancement methods side by side ===
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

image_rel = "142_out.png"   # change to any filename in images/

def load_gray_flat(image_rel, base_dir=base_dir):
    p = Path(base_dir) / "images" / image_rel
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert g is not None, f"Cannot read {p}"
    return g

gray = load_gray_flat(image_rel)

methods = [
    ("Original", lambda g: g),
    ("HE", enhance_he),
    ("CLAHE", lambda g: enhance_clahe(g, clip=2.0, grid=(8,8))),
    ("RSWHE", lambda g: enhance_rswhe(g, recursion=2, alpha=0.6)),
    ("CLAHE_MASK", lambda g: enhance_clahe_masked(g, clip=2.0, grid=(8,8), mask_k=2, strength=0.85)),
    ("RSWHE_MASK", lambda g: enhance_rswhe_masked(g, recursion=2, alpha=0.6, mask_k=2, strength=0.85)),
]

fig, axes = plt.subplots(1, len(methods), figsize=(3.2*len(methods), 3))

for ax, (name, fn) in zip(axes, methods):
    out = fn(gray)
    ax.imshow(out, cmap="gray", vmin=0, vmax=255)
    ax.set_title(name)
    ax.axis("off")

plt.suptitle(f"Contrast Enhancement Comparison: {image_rel}", y=1.05)
plt.tight_layout()
plt.show()


# %%
single_image_table_from_dfs("142_out.png")

# %%
import pandas as pd

def single_image_table_from_dfs(image_rel):
    dfs = [
        ("baseline", df_base),
        ("HE", df_he),
        ("CLAHE", df_clahe),
        ("RSWHE", df_rswhe),
        ("CLAHE_MASK", df_cm),
        ("RSWHE_MASK", df_rm),
    ]

    rows = []
    for name, df in dfs:
        r = df[df["image_rel"] == image_rel]
        if r.empty:
            continue
        r = r.iloc[0].copy()
        r["method"] = name

        # Ensure accuracy exists (compute from pred/gt if needed)
        if "accuracy" not in r.index and ("pred" in r.index) and ("gt" in r.index):
            r["accuracy"] = int(str(r["pred"]) == str(r["gt"]))

        rows.append(r)

    out = pd.DataFrame(rows)

    # Keep only useful columns (drop anything extra your df contains)
    keep = ["method", "accuracy", "ned", "confidence",
            "contrast_pr", "contrast_gain", "delta_mu", "pred", "gt"]
    keep = [c for c in keep if c in out.columns]
    return out[keep].reset_index(drop=True)

# usage
single_image_table_from_dfs("402_3.png")


# %%
single_image_table_from_dfs("142_out.png")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_distribution(metric, title=None, xlim=None):
    dfs = {
        "baseline": df_base,
        "HE": df_he,
        "CLAHE": df_clahe,
        "RSWHE": df_rswhe,
        "CLAHE_MASK": df_cm,
        "RSWHE_MASK": df_rm,
    }

    plt.figure(figsize=(7,5))

    for name, df in dfs.items():
        sns.kdeplot(
            df[metric],
            label=name,
            fill=True,
            alpha=0.25,
            linewidth=2
        )

    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Density")
    plt.title(title if title else f"{metric.upper()} Distribution by Method")
    plt.legend()
    plt.grid(True)

    if xlim:
        plt.xlim(xlim)

    out = fig_dir / f"dist_{metric}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


# %%
plot_metric_distribution("accuracy", title="Accuracy Distribution (Exact Match)", xlim=(0,1))
plot_metric_distribution("ned", title="NED Distribution", xlim=(0,1))
plot_metric_distribution("confidence", title="Confidence Distribution", xlim=(0,1))
plot_metric_distribution("contrast_pr", title="Contrast Level Distribution")
plot_metric_distribution("contrast_gain", title="Contrast Gain Distribution")
plot_metric_distribution("delta_mu", title="Brightness Drift Δμ Distribution")


# %%
import pandas as pd
from pathlib import Path

# 1) Paths (change if your structure is different)
csv_feat = "result/None-VGG-None-CTC.pth/D_prime_combined.csv"
csv_base = "datasets/D_prime/results_baseline.csv"
csv_out  = "datasets/D_prime/results_baseline_with_features.csv"

# 2) Load both tables
df_feat = pd.read_csv(csv_feat)
df_base = pd.read_csv(csv_base)

print("feat columns :", df_feat.columns.tolist())
print("base columns :", df_base.columns.tolist())

# 3) Build a common key using just the filename (no directories)
def to_key(p):
    # handles 'images/402_3.png', './images\\402_3.png', '402_3.png', etc.
    return Path(str(p)).name

df_feat["key"] = df_feat["filename"].apply(to_key)
df_base["key"] = df_base["image_rel"].apply(to_key)

# 4) Merge all columns. We keep *all* columns from both.
df_merged = df_base.merge(
    df_feat,
    on="key",
    how="inner",
    suffixes=("_basecsv", "_featcsv")
)

# (optional) drop duplicate/confusing columns if you want
# e.g., you already have gt/pred/confidence/ned/accuracy in df_base, so you might
# keep the baseline ones and drop the feature-csv duplicates:
drop_cols = [
    "ground_truth", "prediction", "confidence_featcsv",
    "correct", "norm_edit_distance"
]
drop_cols = [c for c in drop_cols if c in df_merged.columns]
df_merged = df_merged.drop(columns=drop_cols)

# 5) Save to a single master CSV
df_merged.to_csv(csv_out, index=False)
print("Saved combined CSV to:", csv_out)
print("Combined columns:", df_merged.columns.tolist())
print("Combined rows   :", len(df_merged))


# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASELINE_CSV = "datasets/D_prime/results_baseline_with_features.csv"
METHODS = ["baseline", "HE", "CLAHE", "RSWHE", "CLAHE_MASK", "RSWHE_MASK"]
FEATURES = [
    "sharp_score",
    "low_light_score",
    "contrast_score",
    "noise_score",
    "sigma",
    "ssim",
    "tenengrad",
    "contrast_pr",
    "delta_mu",
]

# --- load baseline with features ---
df_base_all = pd.read_csv(BASELINE_CSV)

# baseline perf table
base_perf = df_base_all[["image_rel", "accuracy", "ned"]].copy()
base_perf.columns = ["image_rel", "accuracy_base", "ned_base"]

method_dfs = {}

# 1) baseline row: no "helped" (no improvement vs itself)
dfm_base = df_base_all[["image_rel", "accuracy", "ned"] + FEATURES].copy()
dfm_base["accuracy_base"] = dfm_base["accuracy"]
dfm_base["ned_base"] = dfm_base["ned"]
dfm_base["accuracy_m"] = dfm_base["accuracy"]   # same
dfm_base["ned_m"] = dfm_base["ned"]            # same
dfm_base["helped_acc"] = False
dfm_base["helped_ned"] = False
dfm_base["helped_any"] = False
dfm_base["method"] = "baseline"
method_dfs["baseline"] = dfm_base

# 2) enhancement methods
for m in METHODS:
    if m == "baseline":
        continue

    csv_path = f"datasets/D_prime_{m}/results_{m}.csv"
    df_m_raw = pd.read_csv(csv_path)

    # method performance
    df_m_perf = df_m_raw[["image_rel", "accuracy", "ned"]].copy()
    df_m_perf.columns = ["image_rel", "accuracy_m", "ned_m"]

    # merge baseline perf + method perf + features
    dfm = (
        base_perf.merge(df_m_perf, on="image_rel", how="inner")
                 .merge(df_base_all[["image_rel"] + FEATURES], on="image_rel", how="inner")
    )

    # flags
    dfm["helped_acc"] = dfm["accuracy_m"] > dfm["accuracy_base"]
    dfm["helped_ned"] = dfm["ned_m"] > dfm["ned_base"]
    dfm["helped_any"] = dfm["helped_acc"] | dfm["helped_ned"]

    dfm["method"] = m
    method_dfs[m] = dfm

for m, dfm in method_dfs.items():
    print(
        m,
        "rows:", len(dfm),
        "helped_acc:", dfm["helped_acc"].sum(),
        "helped_ned:", dfm["helped_ned"].sum(),
        "helped_any:", dfm["helped_any"].sum(),
    )


# %%
import pandas as pd

BASELINE_CSV = "datasets/D_prime/results_baseline_with_features.csv"
METHODS = ["HE", "CLAHE", "RSWHE", "CLAHE_MASK", "RSWHE_MASK"]

# Load baseline performance (reference)
df_base = pd.read_csv(BASELINE_CSV)[["image_rel", "accuracy", "ned"]].copy()
df_base = df_base.rename(columns={"accuracy": "accuracy_base", "ned": "ned_base"})

acc_rows = []
ned_rows = []

for m in METHODS:
    csv_path = f"datasets/D_prime_{m}/results_{m}.csv"
    df_m = pd.read_csv(csv_path)[["image_rel", "accuracy", "ned"]].copy()
    df_m = df_m.rename(columns={"accuracy": "accuracy_m", "ned": "ned_m"})

    # Merge on image_rel so we compare the same images
    df = df_base.merge(df_m, on="image_rel", how="inner")
    total = len(df)

    # --- Accuracy comparison ---
    acc_better = (df["accuracy_m"] > df["accuracy_base"]).sum()
    acc_same   = (df["accuracy_m"] == df["accuracy_base"]).sum()
    acc_worse  = (df["accuracy_m"] < df["accuracy_base"]).sum()
    acc_better_same = acc_better + acc_same

    acc_rows.append({
        "method": m,
        "same":   acc_same,
        "better": acc_better,
        "worse":  acc_worse,
        "better_same": acc_better_same,  # new column
        "total":  total
    })

    # --- NED comparison ---
    ned_better = (df["ned_m"] > df["ned_base"]).sum()
    ned_same   = (df["ned_m"] == df["ned_base"]).sum()
    ned_worse  = (df["ned_m"] < df["ned_base"]).sum()
    ned_better_same = ned_better + ned_same

    ned_rows.append({
        "method": m,
        "same":   ned_same,
        "better": ned_better,
        "worse":  ned_worse,
        "better_same": ned_better_same,  # new column
        "total":  total
    })

acc_cmp = pd.DataFrame(acc_rows)
ned_cmp = pd.DataFrame(ned_rows)

print("=== Accuracy vs baseline (counts) ===")
display(acc_cmp)

print("=== NED vs baseline (counts) ===")
display(ned_cmp)


# %%
import pandas as pd
import matplotlib.pyplot as plt

BASELINE_CSV = "datasets/D_prime/results_baseline_with_features.csv"
METHODS = ["baseline", "HE", "CLAHE", "RSWHE", "CLAHE_MASK", "RSWHE_MASK"]

# Use only meaningful pre-enhancement features
FEATURES = [
    "sharp_score",
    "low_light_score",
    "contrast_score",
    "noise_score",
    "sigma",
    "tenengrad",
    "contrast_pr",
]

df_base_all = pd.read_csv(BASELINE_CSV)

# baseline perf
base_perf = df_base_all[["image_rel", "accuracy", "ned"]].copy()
base_perf.columns = ["image_rel", "accuracy_base", "ned_base"]

# global baseline feature distribution (same for all rows)
df_baseline_features = df_base_all[["image_rel"] + FEATURES].copy()

method_dfs = {}

# Row 1: baseline (no “improvement” vs itself)
dfm_base = df_baseline_features.copy()
dfm_base["helped_any"] = False
dfm_base["method"] = "baseline"
method_dfs["baseline"] = dfm_base

# Rows 2–6: enhancement methods
for m in METHODS:
    if m == "baseline":
        continue

    csv_path = f"datasets/D_prime_{m}/results_{m}.csv"
    df_m_raw = pd.read_csv(csv_path)

    # method performance
    df_m_perf = df_m_raw[["image_rel", "accuracy", "ned"]].copy()
    df_m_perf.columns = ["image_rel", "accuracy_m", "ned_m"]

    # merge baseline perf + method perf + features
    dfm = (
        base_perf.merge(df_m_perf, on="image_rel", how="inner")
                 .merge(df_baseline_features, on="image_rel", how="inner")
    )

    # improvement criteria: accuracy OR NED improved
    helped_acc = dfm["accuracy_m"] > dfm["accuracy_base"]
    helped_ned = dfm["ned_m"] > dfm["ned_base"]
    dfm["helped_any"] = helped_acc | helped_ned

    # keep only *improved* images for this method
    dfm_improved = dfm[dfm["helped_any"]].copy()
    dfm_improved["method"] = m

    method_dfs[m] = dfm_improved

for m, dfm in method_dfs.items():
    print(m, " rows:", len(dfm))


# %%
n_rows = len(METHODS)
n_cols = len(FEATURES)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3 * n_cols, 2.3 * n_rows),
    sharex="col"
)

if n_rows == 1:
    axes = axes.reshape(1, -1)

for i, method in enumerate(METHODS):
    df_help = method_dfs[method]             # improved images for this method (or all 600 baseline)
    n_help = len(df_help)

    for j, feat in enumerate(FEATURES):
        ax = axes[i, j]

        # baseline distribution (same for all rows)
        all_vals = df_baseline_features[feat].dropna()

        # improved subset for this method
        help_vals = df_help[feat].dropna() if method != "baseline" else []

        ax.hist(all_vals, bins=30, alpha=0.4, label="Baseline all", color="C0")

        if method != "baseline" and len(help_vals) > 0:
            ax.hist(help_vals, bins=30, alpha=0.7, label="Improved images", color="C1")

        if i == 0:
            ax.set_title(feat, fontsize=9)
        if j == 0:
            if method == "baseline":
                ax.set_ylabel(f"{method}\n(N=600)", fontsize=9)
            else:
                ax.set_ylabel(f"{method}\n(improved N={n_help})", fontsize=9)

        ax.tick_params(axis="both", labelsize=7)

# legend from the first non-empty axis
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=9)

plt.tight_layout(rect=(0, 0, 0.95, 1))
plt.show()


# %% [markdown]
# 


