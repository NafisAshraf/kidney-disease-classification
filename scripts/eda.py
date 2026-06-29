from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATA_DIR = Path("artifacts/data_ingestion/kidney-ct-scan-image")
OUTPUT_DIR = Path("assets/eda")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images():
    records = []
    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((path, class_dir.name))
    return records


def image_array(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def save_class_distribution(records):
    counts = {}
    for _, label in records:
        counts[label] = counts.get(label, 0) + 1

    labels = list(counts.keys())
    values = [counts[label] for label in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#0f766e", "#2563eb"], edgecolor="#14213d", linewidth=0.8)
    plt.title("Class Distribution", fontsize=16, weight="bold")
    plt.ylabel("Number of CT Images")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha="center", weight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=180)
    plt.close()

    return counts


def save_sample_grid(records):
    by_class = {}
    for path, label in records:
        by_class.setdefault(label, []).append(path)

    rows = []
    labels = sorted(by_class)
    for label in labels:
        paths = by_class[label]
        indices = np.linspace(0, len(paths) - 1, 4, dtype=int)
        rows.append([paths[index] for index in indices])

    fig, axes = plt.subplots(len(labels), 4, figsize=(12, 7))
    if len(labels) == 1:
        axes = np.array([axes])

    for row_index, label in enumerate(labels):
        for col_index, path in enumerate(rows[row_index]):
            ax = axes[row_index, col_index]
            ax.imshow(image_array(path))
            ax.set_axis_off()
            if col_index == 0:
                ax.text(
                    -0.08,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    fontsize=14,
                    weight="bold",
                    rotation=90,
                    va="center",
                    ha="center",
                )

    fig.suptitle("Representative CT Images by Class", fontsize=16, weight="bold")
    plt.tight_layout(rect=(0.02, 0, 1, 0.94))
    plt.savefig(OUTPUT_DIR / "sample_images_grid.png", dpi=180)
    plt.close()


def build_numeric_profile(records):
    profile = []
    for path, label in records:
        arr = image_array(path)
        gray = np.mean(arr, axis=2)
        profile.append(
            {
                "label": label,
                "width": int(arr.shape[1]),
                "height": int(arr.shape[0]),
                "brightness": float(gray.mean()),
                "contrast": float(gray.std()),
                "red": float(arr[:, :, 0].mean()),
                "green": float(arr[:, :, 1].mean()),
                "blue": float(arr[:, :, 2].mean()),
            }
        )
    return profile


def save_dimension_plot(profile):
    labels = sorted({item["label"] for item in profile})
    widths = [item["width"] for item in profile]
    heights = [item["height"] for item in profile]

    plt.figure(figsize=(7, 6))
    for label in labels:
        x = [item["width"] for item in profile if item["label"] == label]
        y = [item["height"] for item in profile if item["label"] == label]
        plt.scatter(x, y, label=label, alpha=0.72, s=42)
    plt.title("Image Dimension Consistency", fontsize=16, weight="bold")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "image_dimensions.png", dpi=180)
    plt.close()

    return {
        "width_min": int(min(widths)),
        "width_max": int(max(widths)),
        "height_min": int(min(heights)),
        "height_max": int(max(heights)),
    }


def save_brightness_contrast_plot(profile):
    labels = sorted({item["label"] for item in profile})
    brightness = [[item["brightness"] for item in profile if item["label"] == label] for label in labels]
    contrast = [[item["contrast"] for item in profile if item["label"] == label] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(brightness, tick_labels=labels, patch_artist=True)
    axes[0].set_title("Brightness Distribution", fontsize=14, weight="bold")
    axes[0].set_ylabel("Mean grayscale intensity")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(contrast, tick_labels=labels, patch_artist=True)
    axes[1].set_title("Contrast Distribution", fontsize=14, weight="bold")
    axes[1].set_ylabel("Grayscale standard deviation")
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "brightness_contrast.png", dpi=180)
    plt.close()


def save_channel_mean_plot(profile):
    labels = sorted({item["label"] for item in profile})
    channels = ["red", "green", "blue"]
    channel_colors = ["#dc2626", "#16a34a", "#2563eb"]

    values = []
    for label in labels:
        label_items = [item for item in profile if item["label"] == label]
        values.append([np.mean([item[channel] for item in label_items]) for channel in channels])

    x = np.arange(len(labels))
    width = 0.22

    plt.figure(figsize=(9, 5))
    for index, channel in enumerate(channels):
        plt.bar(
            x + (index - 1) * width,
            [row[index] for row in values],
            width=width,
            label=channel.title(),
            color=channel_colors[index],
        )
    plt.title("Average RGB Channel Intensity", fontsize=16, weight="bold")
    plt.ylabel("Mean pixel value")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "channel_means.png", dpi=180)
    plt.close()


def save_summary(counts, dimensions, profile):
    labels = sorted(counts)
    summary = {
        "total_images": len(profile),
        "class_counts": counts,
        "dimensions": dimensions,
        "mean_brightness_by_class": {
            label: round(float(np.mean([item["brightness"] for item in profile if item["label"] == label])), 2)
            for label in labels
        },
        "mean_contrast_by_class": {
            label: round(float(np.mean([item["contrast"] for item in profile if item["label"] == label])), 2)
            for label in labels
        },
    }
    with open(OUTPUT_DIR / "dataset_profile.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_images()
    if not records:
        raise FileNotFoundError(f"No images found in {DATA_DIR}")

    counts = save_class_distribution(records)
    save_sample_grid(records)
    profile = build_numeric_profile(records)
    dimensions = save_dimension_plot(profile)
    save_brightness_contrast_plot(profile)
    save_channel_mean_plot(profile)
    summary = save_summary(counts, dimensions, profile)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
