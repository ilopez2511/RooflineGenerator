
import sys
import torch
import matplotlib.pyplot as plt
import math
from matplotlib.legend import Legend
from torch import nn
from ModelProfiling import (
    init_cypapi,
    get_default_nvidia_events,
    run_profiling,
    summarize_oi_perf
)

from rooflineModel import Roofline

sys.path.append(
    "/global/homes/i/irvinl25/hpc_results_v3.0/"
    "HPE+LBNL/benchmarks/deepcam/implementations/"
    "deepcam-pytorch/src/deepCam/architecture"
)
from deeplab_xception import DeepLabv3_plus
from torchvision.models import (resnet101, mobilenet_v2, efficientnet_b0)

def build_resnet_model(device="cuda"):
    model = resnet101(weights=None).train().to(device)
    x = torch.randn(64, 3, 224, 224, device=device, requires_grad=True)
    return model, x, "ResNet101"


def build_mobilenet(device="cuda"):
    model = mobilenet_v2(weights=None).train().to(device)
    x = torch.randn(64, 3, 224, 224, device=device, requires_grad=True)
    return model, x, "MobileNetV2"


def build_efficientnet(device="cuda"):
    model = efficientnet_b0(weights=None).train().to(device)
    x = torch.randn(64, 3, 224, 224, device=device, requires_grad=True)
    return model, x, "EfficientNetB0"


def build_deeplab(device="cuda"):
    model = DeepLabv3_plus(n_input=16, n_classes=3).train().to(device)
    x = torch.randn(64, 16, 256, 256, device=device, requires_grad=True)
    return model, x, "DeepLabv3+"

class DummyGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

USE_MODEL = "deeplab"  # "deeplab", "resnet", "efficientnet"

def build_model_and_input(device="cuda"):
    if USE_MODEL == "deeplab":
        return build_deeplab(device)
    if USE_MODEL == "resnet":
        return build_resnet_model(device)
    if USE_MODEL == "mobilenet":
        return build_mobilenet(device)
    if USE_MODEL == "efficientnet":
        return build_efficientnet(device)

    raise ValueError("Unknown model name")


A100_FP32_PEAK = 19.5e12      
A100_FP16_TENSOR_PEAK = 312e12  
A100_DRAM_BW_BYTES = 1.935e12   

def generate_roofline_from_summary(summary: dict):

    label = summary.get("label", "Model")
    safe_label = label.replace(" ", "_")

    pi = [
        ("CUDA_FP32_peak", 0.0, A100_FP32_PEAK),
        ("Tensor_FP16_peak", 0.0, A100_FP16_TENSOR_PEAK),
    ]

    
    betas = [
        ("DRAM_BW", A100_DRAM_BW_BYTES, 0.0),
    ] 

    oi_vals = []
    for side in ("fwd", "bwd"):
        for engine in ("cuda", "tensor", "total"):
            oi = summary.get(side, {}).get(engine, {}).get("oi", None)
            if oi is not None and oi > 0:
                oi_vals.append(oi)

    if not oi_vals:
        # Fallback if something weird happens
        oi_min, oi_max = 1e0, 1e2
    else:
        oi_min = min(oi_vals)
        oi_max = max(oi_vals)

    # --- Simple log-based padding: one decade below min, one above max ---
    min_log = math.floor(math.log10(oi_min))
    max_log = math.ceil(math.log10(oi_max))

    x_min = 10 ** (min_log - 1)   # one decade below lowest OI
    x_max = 10 ** (max_log + 1)   # one decade above highest OI

    # Optionally clamp so it never goes below 1e-2, etc.
    x_min = max(x_min, 1e-2)

    # Keep your existing y-range or tweak similarly if you want
    y_min, y_max = 1e11, 1e15

    rf = Roofline(
        pi,
        betas,
        "Operational Intensity (FLOPs/Byte)",
        "Performance (FLOP/s)",
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
    )

    fwd = summary["fwd"]
    bwd = summary["bwd"]

    data = {
        f"{label} FWD CUDA":   {"TOTAL": (fwd["cuda"]["oi"],   fwd["cuda"]["perf"])},
        f"{label} FWD TENSOR": {"TOTAL": (fwd["tensor"]["oi"], fwd["tensor"]["perf"])},
        f"{label} FWD TOTAL":  {"TOTAL": (fwd["total"]["oi"],  fwd["total"]["perf"])},

        f"{label} BWD CUDA":   {"TOTAL": (bwd["cuda"]["oi"],   bwd["cuda"]["perf"])},
        f"{label} BWD TENSOR": {"TOTAL": (bwd["tensor"]["oi"], bwd["tensor"]["perf"])},
        f"{label} BWD TOTAL":  {"TOTAL": (bwd["total"]["oi"],  bwd["total"]["perf"])},
    }

    rf.addLabelData(data)
    rf.plot()

    ax = plt.gca()

    for txt in list(ax.texts):
        if "TOTAL" in txt.get_text():
            txt.remove()

    handles, labels = ax.get_legend_handles_labels()

    for artist in list(ax.get_children()):
        if isinstance(artist, Legend):
            artist.set_visible(False)

    ax.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()
    out_path = f"roofline_{safe_label}.png"

    fig = plt.gcf()
    fig.set_size_inches(14, 9)   # large, thesis-grade image

    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved roofline plot to {out_path}")


def main():
    init_cypapi()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, x, label = build_model_and_input(device=device)

    profiling_events = get_default_nvidia_events()

    # 1) Profiling → DataFrame
    df = run_profiling(
        model,
        x,
        profiling_events,
        num_runs=50,
        warmup=5,
        label=label,
    )

    # 2) Save CSV/JSON as before
    safe_label = label.replace(" ", "_")
    df.to_csv(f"profile_{safe_label}.csv", index=False)
    df.to_json(f"profile_{safe_label}.json", orient="records", indent=2)
    print(f"\nSaved: profile_{safe_label}.csv and profile_{safe_label}.json")

    # 3) Build a summary dict for easy consumption
    summary = summarize_oi_perf(df, label=label)
    print("\nSummary dict for roofline:")
    print(summary)

    # 4) Generate the roofline plot (start with just DeepLab / current model)
    generate_roofline_from_summary(summary)


if __name__ == "__main__":
    main()
