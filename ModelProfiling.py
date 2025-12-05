import torch
import cypapi as cyp
import pandas as pd
from torch.amp import autocast

# Global PyTorch settings (good for A100, TF32 enabled kernels)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False)

DEFAULT_SEED = 1234


def init_cypapi():
    """Initialize cyPAPI once before profiling."""
    cyp.cyPAPI_library_init(cyp.PAPI_VER_CURRENT)
    if cyp.cyPAPI_is_initialized() != 1:
        raise RuntimeError("cyPAPI init failed")


def get_default_nvidia_events():
    """
    Default CUDA events for NVIDIA GPUs.
    No device suffix -- CUPTI auto-selects active GPU.
    """
    return {
        "ffma": "cuda:::smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "fadd": "cuda:::smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "fmul": "cuda:::smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
        "tensor_fp16_to_fp32": "cuda:::sm__ops_path_tensor_src_fp16_dst_fp32.sum",
        "tensor_tf32_to_fp32": "cuda:::sm__ops_path_tensor_src_tf32_dst_fp32.sum",
        "l1_bytes": "cuda:::l1tex__t_bytes.sum",
        "l2_bytes": "cuda:::lts__t_bytes.sum",
        "dram_bytes_read": "cuda:::dram__bytes_read.sum",
        "dram_bytes_write": "cuda:::dram__bytes_write.sum",
    }


def set_seed(seed=DEFAULT_SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_forward(model, x) -> float:
    """Measure forward time once (ms) with no counters."""
    _sync()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    if torch.cuda.is_available():
        with autocast(device_type="cuda"):
            _ = model(x)
    else:
        _ = model(x)
    end.record()
    _sync()
    return start.elapsed_time(end)


def time_backward(model, x) -> float:
    #Measure backward time once (ms) with no counters.
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        with autocast(device_type="cuda"):
            out = model(x)
            loss = out.sum()
    else:
        out = model(x)
        loss = out.sum()

    _sync()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    loss.backward()
    end.record()
    _sync()
    return start.elapsed_time(end)


def count_event_forward(event_name, event_code, model, x) -> float:
    #Run one forward pass and count a single CUDA event.
    es = cyp.CypapiCreateEventset()
    es.add_event(event_code)
    _sync()
    es.start()
    if torch.cuda.is_available():
        with autocast(device_type="cuda"):
            _ = model(x)
    else:
        _ = model(x)
    es.stop()
    vals = es.read()
    es.reset()
    return float(vals[0])


def count_event_backward(event_name, event_code, model, x) -> float:
    #Run one backward pass (fresh graph) and count a single event.
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        with autocast(device_type="cuda"):
            out = model(x)
            loss = out.sum()
    else:
        out = model(x)
        loss = out.sum()

    es = cyp.CypapiCreateEventset()
    es.add_event(event_code)
    _sync()
    es.start()
    loss.backward()
    es.stop()
    vals = es.read()
    es.reset()
    return float(vals[0])


def run_profiling(model, x, profiling_events, num_runs=50, warmup=5, label=None):
    """
    Profile forward & backward passes using cyPAPI metrics.
    Returns a DataFrame with all raw and derived metrics.
    """

    # Warmup
    for _ in range(warmup):
        _ = time_forward(model, x)
        _ = time_backward(model, x)

    rows = []
    for i in range(num_runs):
        set_seed(DEFAULT_SEED + i)

        # timing passes
        fwd_ms = time_forward(model, x)
        bwd_ms = time_backward(model, x)

        row = {"run": i, "fwd_time_ms": fwd_ms, "bwd_time_ms": bwd_ms}
        if label is not None:
            row["model_label"] = label

        # per-event counters
        for name, event_string in profiling_events.items():
            try:
                code = cyp.cyPAPI_event_name_to_code(event_string)
            except Exception as e:
                print(f"[WARN] cannot register {name}: {e}")
                row[f"fwd_{name}"] = float("nan")
                row[f"bwd_{name}"] = float("nan")
                continue

            set_seed(DEFAULT_SEED + i)
            row[f"fwd_{name}"] = count_event_forward(name, code, model, x)

            set_seed(DEFAULT_SEED + i)
            row[f"bwd_{name}"] = count_event_backward(name, code, model, x)

        rows.append(row)

    df = pd.DataFrame(rows)

    # derived metrics (FLOPs, Bytes, OI, Perf)
    def flops_cuda(prefix):
        return (
            2 * df.get(f"{prefix}_ffma", 0).fillna(0)
            + df.get(f"{prefix}_fadd", 0).fillna(0)
            + df.get(f"{prefix}_fmul", 0).fillna(0)
        )

    def flops_tensor(prefix):
        return (
            df.get(f"{prefix}_tensor_fp16_to_fp32", 0).fillna(0)
            + df.get(f"{prefix}_tensor_tf32_to_fp32", 0).fillna(0)
        )

    for p in ["fwd", "bwd"]:
        df[f"{p}_flops_cuda"] = flops_cuda(p)
        df[f"{p}_flops_tensor"] = flops_tensor(p)
        df[f"{p}_total_flops"] = df[f"{p}_flops_cuda"] + df[f"{p}_flops_tensor"]

        df[f"{p}_dram_bytes"] = (
            df.get(f"{p}_dram_bytes_read", 0).fillna(0)
            + df.get(f"{p}_dram_bytes_write", 0).fillna(0)
        )

        time_s = df[f"{p}_time_ms"] / 1000.0
        bytes_ = df[f"{p}_dram_bytes"].replace(0, pd.NA)

        df[f"{p}_oi_cuda"] = df[f"{p}_flops_cuda"] / bytes_
        df[f"{p}_oi_tensor"] = df[f"{p}_flops_tensor"] / bytes_
        df[f"{p}_oi_total"] = df[f"{p}_total_flops"] / bytes_

        df[f"{p}_perf_cuda"] = df[f"{p}_flops_cuda"] / time_s
        df[f"{p}_perf_tensor"] = df[f"{p}_flops_tensor"] / time_s
        df[f"{p}_perf_total"] = df[f"{p}_total_flops"] / time_s

    # summary
    avg = df.mean(numeric_only=True)

    def line(pass_type, engine):
        return (
            f"{pass_type.upper()} {engine.upper()} "
            f"OI={avg.get(f'{pass_type}_oi_{engine}', float('nan')):.4f}  "
            f"Perf={avg.get(f'{pass_type}_perf_{engine}', float('nan')):.3e} FLOP/s"
        )

    print("\n=== Profiling Summary ===")
    if label:
        print(f"Model: {label}")
    print(f"FWD time: {avg['fwd_time_ms']:.3f} ms | BWD time: {avg['bwd_time_ms']:.3f} ms")
    print(f"FWD DRAM: {avg['fwd_dram_bytes']:.3e} B | BWD DRAM: {avg['bwd_dram_bytes']:.3e} B")
    print(line("fwd", "cuda"))
    print(line("fwd", "tensor"))
    print(line("fwd", "total"))
    print(line("bwd", "cuda"))
    print(line("bwd", "tensor"))
    print(line("bwd", "total"))

    return df
    
def summarize_oi_perf(df: pd.DataFrame, label: str | None = None):
    """
    Build a small dict with average OI and Perf for CUDA, Tensor, and Total,
    for both forward and backward passes.

    This is meant to be consumed by the roofline generator.
    """
    avg = df.mean(numeric_only=True)
    if label is None and "model_label" in df.columns:
        label = df["model_label"].iloc[0]

    def block(prefix: str):
        return {
            "cuda": {
                "oi": float(avg.get(f"{prefix}_oi_cuda", float("nan"))),
                "perf": float(avg.get(f"{prefix}_perf_cuda", float("nan"))),
            },
            "tensor": {
                "oi": float(avg.get(f"{prefix}_oi_tensor", float("nan"))),
                "perf": float(avg.get(f"{prefix}_perf_tensor", float("nan"))),
            },
            "total": {
                "oi": float(avg.get(f"{prefix}_oi_total", float("nan"))),
                "perf": float(avg.get(f"{prefix}_perf_total", float("nan"))),
            },
        }

    summary = {
        "label": label,
        "fwd": block("fwd"),
        "bwd": block("bwd"),
    }
    return summary

