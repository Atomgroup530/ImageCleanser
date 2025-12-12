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
    "evaluate_model": True,
    "evaluate_dataset": True,
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
# ## Contrast

# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from nltk.metrics.distance import edit_distance
from IPython.display import HTML, display

import torch
from torch.utils.data import DataLoader

# Clova OCR imports
from dataset import RawDataset, AlignCollate
from model import Model
from utils import CTCLabelConverter
from create_lmdb_dataset import createDataset


# %%
from skimage.metrics import structural_similarity as ssim

def compute_contrast_metrics(gray_enh, gray_orig=None):
    """Contrast σ, SSIM, Tenengrad, Δμ brightness shift."""

    gray_enh = gray_enh.astype(np.uint8)

    # 1) Global contrast = std
    sigma = float(np.std(gray_enh))

    # 2) SSIM
    if gray_orig is None:
        ssim_val = 1.0
    else:
        gray_orig = gray_orig.astype(np.uint8)
        try:
            ssim_val = float(ssim(gray_orig, gray_enh, data_range=255))
        except:
            # fallback for size mismatch
            h = min(gray_orig.shape[0], gray_enh.shape[0])
            w = min(gray_orig.shape[1], gray_enh.shape[1])
            ssim_val = float(ssim(gray_orig[:h,:w], gray_enh[:h,:w], data_range=255))

    # 3) Tenengrad = sum of Sobel magnitude
    gx = cv2.Sobel(gray_enh, cv2.CV_64F, 1,0, ksize=3)
    gy = cv2.Sobel(gray_enh, cv2.CV_64F, 0,1, ksize=3)
    tenengrad = float(np.sum(gx**2 + gy**2))

    # 4) Δμ brightness shift
    if gray_orig is None:
        delta_mu = 0.0
    else:
        delta_mu = abs(float(np.mean(gray_enh)) - float(np.mean(gray_orig)))

    return {
        "sigma": sigma,
        "ssim": ssim_val,
        "tenengrad": tenengrad,
        "delta_mu": delta_mu,
    }


# %%
def enhance_he(gray):
    return cv2.equalizeHist(gray)

def enhance_clahe(gray, clip=2.0, grid=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray)


# %%
from types import SimpleNamespace

opt = SimpleNamespace()

# MODEL
opt.saved_model = "saved_models/None-VGG-None-CTC.pth"   # <-- CHANGE IF NEEDED
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
def run_scoring_and_eval_on_folder(method_name, dataset_root, opt):
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images"
    gt_path = dataset_root / "gt.txt"

    # Read GT
    label_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            rel, label = line.strip().split("\t")
            label_map[rel] = label

    # Data loader
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    raw_dataset = RawDataset(root=str(img_dir), opt=opt)
    loader = DataLoader(raw_dataset, batch_size=192, shuffle=False,
                        num_workers=0, collate_fn=align_collate)

    # Model
    model, converter = load_ocr_model(opt)
    rows = []

    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc=f"[{method_name}] Eval"):
            imgs = imgs.to(device)
            bsz = imgs.size(0)

            len_for_pred = torch.IntTensor([opt.batch_max_length] * bsz).to(device)
            txt_for_pred = torch.zeros(bsz, opt.batch_max_length+1, dtype=torch.long).to(device)

            preds = model(imgs, txt_for_pred)
            preds_prob = torch.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            _, preds_index = preds.max(2)

            preds_str = converter.decode(preds_index, torch.IntTensor([preds.size(1)] * bsz))

            for img_path, pred, prob_seq in zip(paths, preds_str, preds_max_prob):
                rel = os.path.relpath(img_path, start=str(img_dir))
                gt = label_map[rel]

                conf = float(prob_seq.cumprod(dim=0)[-1])

                if len(gt)==0 or len(pred)==0:
                    ned = 0
                elif len(gt)>=len(pred):
                    ned = 1 - edit_distance(pred, gt) / len(gt)
                else:
                    ned = 1 - edit_distance(pred, gt) / len(pred)

                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                dip = compute_contrast_metrics(gray)

                rows.append({
                    "method": method_name,
                    "image_rel": rel,
                    "gt": gt,
                    "pred": pred,
                    "confidence": conf,
                    "ned": ned,
                    "sigma": dip["sigma"],
                    "ssim": dip["ssim"],
                    "tenengrad": dip["tenengrad"],
                    "delta_mu": dip["delta_mu"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(dataset_root / f"results_{method_name}.csv", index=False)
    print(f"Saved: {dataset_root}/results_{method_name}.csv")

    return df


# %%
def get_top10(df_base, df_enh, method):
    merged = df_base.merge(df_enh, on="image_rel", suffixes=("_base", f"_{method}"))
    merged["ned_delta"] = merged[f"ned_{method}"] - merged["ned_base"]
    return merged.sort_values("ned_delta", ascending=False).head(10)


# %%
def show_top10(top10, baseline_root, enh_root, method):
    html = ""
    for _, row in top10.iterrows():
        rel = row["image_rel"]
        b_img = baseline_root / "images" / rel
        e_img = enh_root / "images" / rel

        html += f"""
        <h3>{rel}</h3>
        <table border="1" style="border-collapse: collapse;">
        <tr><th></th><th>Image</th><th>GT / Pred</th><th>NED</th><th>Conf</th><th>σ</th><th>Tenengrad</th><th>Δμ</th></tr>

        <tr>
            <td>Baseline</td>
            <td><img src="{b_img}" width="220"></td>
            <td>{row['gt_base']} / {row['pred_base']}</td>
            <td>{row['ned_base']:.3f}</td>
            <td>{row['confidence_base']:.3f}</td>
            <td>{row['sigma_base']:.3f}</td>
            <td>{row['tenengrad_base']:.3f}</td>
            <td>{row['delta_mu_base']:.3f}</td>
        </tr>

        <tr>
            <td>{method}</td>
            <td><img src="{e_img}" width="220"></td>
            <td>{row[f'gt_{method}']} / {row[f'pred_{method}']}</td>
            <td>{row[f'ned_{method}']:.3f}</td>
            <td>{row[f'confidence_{method}']:.3f}</td>
            <td>{row[f'sigma_{method}']:.3f}</td>
            <td>{row[f'tenengrad_{method}']:.3f}</td>
            <td>{row[f'delta_mu_{method}']:.3f}</td>
        </tr>
        </table><hr/>
        """

    display(HTML(html))


# %%
def plot_distributions(df_base, df_enh, method):
    metrics = ["confidence", "ned", "sigma", "tenengrad", "delta_mu"]
    plt.figure(figsize=(14, 4 * len(metrics)))

    for i, m in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        plt.hist(df_base[m], bins=40, alpha=0.5, label="baseline")
        plt.hist(df_enh[m], bins=40, alpha=0.5, label=method)
        plt.title(f"{m} distribution: baseline vs {method}")
        plt.legend()

    plt.tight_layout()
    plt.show()


# %%
def summarize_methods(dfs):
    rows = []
    for name, df in dfs.items():
        rows.append({
            "method": name,
            "mean_conf": df["confidence"].mean(),
            "mean_ned": df["ned"].mean(),
            "mean_sigma": df["sigma"].mean(),
            "mean_tenengrad": df["tenengrad"].mean(),
            "mean_delta_mu": df["delta_mu"].mean(),
        })
    return pd.DataFrame(rows).set_index("method")


# %%
import shutil
from pathlib import Path

def rebuild_enhanced_dataset(method_name, enhancer_fn, src_root="datasets/D_prime"):
    src_root = Path(src_root)
    src_img_dir = src_root / "images"
    src_gt = src_root / "gt.txt"

    dst_root = src_root.parent / f"D_prime_{method_name}"
    dst_img_dir = dst_root / "images"
    dst_gt = dst_root / "gt.txt"

    # Remove old dataset entirely
    if dst_root.exists():
        shutil.rmtree(dst_root)

    dst_img_dir.mkdir(parents=True, exist_ok=True)

    # ---- Normalize & copy GT ----
    with open(src_gt, "r", encoding="utf-8") as f_in, open(dst_gt, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rel, label = line.strip().split("\t")
            if rel.startswith("images/"):
                rel = rel[7:]
            f_out.write(f"{rel}\t{label}\n")

    # ---- Enhance images ----
    img_paths = list(src_img_dir.glob("*"))
    for p in tqdm(img_paths, desc=f"Building {method_name}"):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        enh = enhancer_fn(gray)
        out_path = dst_img_dir / p.name
        cv2.imwrite(str(out_path), enh)

    print(f"Rebuilt dataset: {dst_root}")
    return dst_root


# %%
he_root = rebuild_enhanced_dataset("HE", enhance_he)
clahe_root = rebuild_enhanced_dataset("CLAHE", enhance_clahe)


# %%
def normalize_all_gt(root):
    gt = Path(root) / "gt.txt"
    if not gt.exists():
        print("No gt.txt:", gt)
        return

    lines = []
    for line in open(gt, "r", encoding="utf-8"):
        img_rel, label = line.strip().split("\t")
        if img_rel.startswith("images/"):
            img_rel = img_rel[7:]   # strip prefix
        lines.append(f"{img_rel}\t{label}\n")

    with open(gt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Normalized:", gt)


normalize_all_gt("datasets/D_prime")
normalize_all_gt("datasets/D_prime_HE")
normalize_all_gt("datasets/D_prime_CLAHE")


# %%
!head datasets/D_prime/gt.txt
!head datasets/D_prime_HE/gt.txt
!head datasets/D_prime_CLAHE/gt.txt


# %%
baseline_root = Path("datasets/D_prime")
he_root       = Path("datasets/D_prime_HE")
clahe_root    = Path("datasets/D_prime_CLAHE")

print("Evaluating baseline...")
df_base = run_scoring_and_eval_on_folder("baseline", baseline_root, opt)

print("Evaluating HE...")
df_he = run_scoring_and_eval_on_folder("HE", he_root, opt)

print("Evaluating CLAHE...")
df_clahe = run_scoring_and_eval_on_folder("CLAHE", clahe_root, opt)

# ---- Top 10 ----
top10_he = get_top10(df_base, df_he, "HE")
top10_clahe = get_top10(df_base, df_clahe, "CLAHE")

show_top10(top10_he, baseline_root, he_root, "HE")
show_top10(top10_clahe, baseline_root, clahe_root, "CLAHE")

# ---- Plots ----
plot_distributions(df_base, df_he, "HE")
plot_distributions(df_base, df_clahe, "CLAHE")

# ---- Summary ----
summary = summarize_methods({
    "baseline": df_base,
    "HE": df_he,
    "CLAHE": df_clahe,
})
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load your CSVs ===
df_base = pd.read_csv("datasets/D_prime/results_baseline.csv")
df_he = pd.read_csv("datasets/D_prime_HE/results_HE.csv")
df_clahe = pd.read_csv("datasets/D_prime_CLAHE/results_CLAHE.csv")

# Combine mean metrics for quick plotting
summary = pd.DataFrame({
    "baseline": {
        "mean_ned": df_base["ned"].mean(),
        "mean_conf": df_base["confidence"].mean(),
        "mean_sigma": df_base["sigma"].mean(),
    },
    "HE": {
        "mean_ned": df_he["ned"].mean(),
        "mean_conf": df_he["confidence"].mean(),
        "mean_sigma": df_he["sigma"].mean(),
    },
    "CLAHE": {
        "mean_ned": df_clahe["ned"].mean(),
        "mean_conf": df_clahe["confidence"].mean(),
        "mean_sigma": df_clahe["sigma"].mean(),
    }
}).T

# ----------------------------
# CHART 1: NED BAR CHART
# ----------------------------
plt.figure(figsize=(6,4))
plt.bar(summary.index, summary["mean_ned"], color=["gray","steelblue","orange"])
plt.title("Mean NED (Baseline vs HE vs CLAHE)")
plt.ylabel("NED")
plt.ylim([0, max(summary["mean_ned"])*1.2])
plt.grid(axis="y", alpha=0.3)
plt.show()

# ----------------------------
# CHART 2: CONFIDENCE BAR CHART
# ----------------------------
plt.figure(figsize=(6,4))
plt.bar(summary.index, summary["mean_conf"], color=["gray","steelblue","orange"])
plt.title("Mean Confidence (Baseline vs HE vs CLAHE)")
plt.ylabel("Confidence")
plt.ylim([0, max(summary["mean_conf"])*1.2])
plt.grid(axis="y", alpha=0.3)
plt.show()

# ----------------------------
# CHART 3: SIGMA BAR CHART (contrast)
# ----------------------------
plt.figure(figsize=(6,4))
plt.bar(summary.index, summary["mean_sigma"], color=["gray","steelblue","orange"])
plt.title("Mean Contrast (Sigma)")
plt.ylabel("Sigma (std of pixel values)")
plt.grid(axis="y", alpha=0.3)
plt.show()

# ----------------------------
# CHART 4: Combined 3×1 Slide Chart
# ----------------------------
fig, axs = plt.subplots(3, 1, figsize=(7,10))

axs[0].bar(summary.index, summary["mean_ned"], color=["gray","steelblue","orange"])
axs[0].set_title("Mean NED")
axs[0].grid(axis="y", alpha=0.3)

axs[1].bar(summary.index, summary["mean_conf"], color=["gray","steelblue","orange"])
axs[1].set_title("Mean Confidence")
axs[1].grid(axis="y", alpha=0.3)

axs[2].bar(summary.index, summary["mean_sigma"], color=["gray","steelblue","orange"])
axs[2].set_title("Mean Contrast (Sigma)")
axs[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load CSVs ===
df_base = pd.read_csv("datasets/D_prime/results_baseline.csv")
df_he = pd.read_csv("datasets/D_prime_HE/results_HE.csv")
df_clahe = pd.read_csv("datasets/D_prime_CLAHE/results_CLAHE.csv")

# Build summary table
summary = pd.DataFrame({
    "baseline": {
        "mean_ned": df_base["ned"].mean(),
        "mean_conf": df_base["confidence"].mean(),
        "mean_sigma": df_base["sigma"].mean(),
    },
    "HE": {
        "mean_ned": df_he["ned"].mean(),
        "mean_conf": df_he["confidence"].mean(),
        "mean_sigma": df_he["sigma"].mean(),
    },
    "CLAHE": {
        "mean_ned": df_clahe["ned"].mean(),
        "mean_conf": df_clahe["confidence"].mean(),
        "mean_sigma": df_clahe["sigma"].mean(),
    }
}).T

methods = summary.index.tolist()

# A helper to draw a bar chart WITH numbers on bars
def bar_with_numbers(values, title, ylabel):
    plt.figure(figsize=(6,4))
    bars = plt.bar(methods, values, color=["gray","steelblue","orange"])

    # Add numeric labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + (0.02 * height),
            f"{height:.4f}",
            ha='center', va='bottom', fontsize=10
        )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Plotting ---
bar_with_numbers(summary["mean_ned"],  "Mean NED (Baseline vs HE vs CLAHE)", "NED")
bar_with_numbers(summary["mean_conf"], "Mean Confidence", "Confidence")
bar_with_numbers(summary["mean_sigma"], "Mean Contrast (Sigma)", "Sigma")



