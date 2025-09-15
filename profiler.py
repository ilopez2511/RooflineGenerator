import torch
import torch.nn as nn
import cypapi as cyp
import pandas as pd
import time
from torch.amp import autocast
from torchvision.models import resnet101

import os
import sys

# append the directory where deeplab_xception.py lives
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir))

# sys.path.append("/global/homes/i/irvinl25/hpc_results_v3.0/HPE+LBNL/benchmarks/deepcam/implementations/deepcam-pytorch/src/deepCam/architecture")
from deeplab_xception import DeepLabv3_plus

def profile_forward_pass(event_codes, model, input_tensor):
    metrics = {}
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()

    for event_name, event_code in event_codes.items():
        eventset = cyp.CypapiCreateEventset()
        eventset.add_event(event_code)
        eventset.start()
        with autocast(device_type='cuda'):
            output = model(input_tensor)
        eventset.stop()
        metrics[event_name] = eventset.read()[0]
        eventset.reset()

    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    return metrics, elapsed_ms

def profile_backward_pass(event_codes, model, input_tensor):
    with autocast(device_type='cuda'):
        output = model(input_tensor)
    loss = output.sum()

    metrics = {}
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()

    for event_name, event_code in event_codes.items():
        eventset = cyp.CypapiCreateEventset()
        eventset.add_event(event_code)
        eventset.start()
        loss.backward(retain_graph=True)
        eventset.stop()
        metrics[event_name] = eventset.read()[0]
        eventset.reset()

    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    return metrics, elapsed_ms

def run_combined_profiling(event_codes, model, input_tensor, num_runs):
    results = {
        "forward_metrics": {name: [] for name in event_codes},
        "backward_metrics": {name: [] for name in event_codes},
        "forward_times": [],
        "backward_times": []
    }

    for _ in range(num_runs):
        fwd_metrics, fwd_time = profile_forward_pass(event_codes, model, input_tensor)
        bwd_metrics, bwd_time = profile_backward_pass(event_codes, model, input_tensor)

        for name in event_codes:
            results["forward_metrics"][name].append(fwd_metrics[name])
            results["backward_metrics"][name].append(bwd_metrics[name])

        results["forward_times"].append(fwd_time)
        results["backward_times"].append(bwd_time)

    return results

if __name__ == "__main__":
    cyp.cyPAPI_library_init(cyp.PAPI_VER_CURRENT)

    if cyp.cyPAPI_is_initialized() != 1:
        raise ValueError("cyPAPI has not been initialized.\n")

    profiling_events = {
        "ffma": "cuda:::smsp__sass_thread_inst_executed_op_ffma_pred_on.sum:device=0",
        "fadd": "cuda:::smsp__sass_thread_inst_executed_op_fadd_pred_on.sum:device=0",
        "fmul": "cuda:::smsp__sass_thread_inst_executed_op_fmul_pred_on.sum:device=0",
        "tensor_fp16_to_fp32": "cuda:::sm__ops_path_tensor_src_fp16_dst_fp32.sum:device=0",
        "tensor_tf32_to_fp32": "cuda:::sm__ops_path_tensor_src_tf32_dst_fp32.sum:device=0",
        "l1_bytes": "cuda:::l1tex__t_bytes.sum:device=0",
        "l2_bytes": "cuda:::lts__t_bytes.sum:device=0",
        "dram_bytes_read": "cuda:::dram__bytes_read.sum:device=0",
        "dram_bytes_write": "cuda:::dram__bytes_write.sum:device=0"
    }

    event_codes = {}
    for name, event in profiling_events.items():
        try:
            event_codes[name] = cyp.cyPAPI_event_name_to_code(event)
        except Exception as e:
            print(f"Failed to register event '{name}': {e}")

    model = DeepLabv3_plus(n_input=16, n_classes=3)
    model.train()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    input_tensor = torch.randn(64, 16, 256, 256).to(device)
    input_tensor.requires_grad = True

    results = run_combined_profiling(event_codes, model, input_tensor, num_runs=50)

    df = pd.DataFrame([
        {
            "run": i,
            "fwd_time_ms": results["forward_times"][i],
            "bwd_time_ms": results["backward_times"][i],
            **{f"fwd_{k}": results["forward_metrics"][k][i] for k in event_codes},
            **{f"bwd_{k}": results["backward_metrics"][k][i] for k in event_codes},
            "fwd_total_flops": (
                2 * results["forward_metrics"].get("ffma", [0])[i] +
                results["forward_metrics"].get("fadd", [0])[i] +
                results["forward_metrics"].get("fmul", [0])[i] +
                results["forward_metrics"].get("tensor_fp16_to_fp32", [0])[i] +
                results["forward_metrics"].get("tensor_tf32_to_fp32", [0])[i]
            ),
            "bwd_total_flops": (
                2 * results["backward_metrics"].get("ffma", [0])[i] +
                results["backward_metrics"].get("fadd", [0])[i] +
                results["backward_metrics"].get("fmul", [0])[i] +
                results["backward_metrics"].get("tensor_fp16_to_fp32", [0])[i] +
                results["backward_metrics"].get("tensor_tf32_to_fp32", [0])[i]
            ),
            "fwd_dram_bytes": results["forward_metrics"].get("dram_bytes_read", [0])[i] + results["forward_metrics"].get("dram_bytes_write", [0])[i],
            "bwd_dram_bytes": results["backward_metrics"].get("dram_bytes_read", [0])[i] + results["backward_metrics"].get("dram_bytes_write", [0])[i]
        }
        for i in range(len(results["forward_times"]))
    ])

    df.to_csv("profiling_results_split_passes.csv", index=False)
    df.to_json("profiling_results_split_passes.json", orient="records", indent=4)

    print("\nProfiling results saved to 'profiling_results_split_passes.csv' and '.json'.")
    print(df.head())

    averages = df.mean()

    print("\n--- Roofline Coordinates ---")
    for pass_type in ["fwd", "bwd"]:
        flops = averages[f"{pass_type}_total_flops"]
        dram = averages[f"{pass_type}_dram_bytes"]
        time_sec = averages[f"{pass_type}_time_ms"] / 1000
        oi = flops / dram
        perf = flops / time_sec
        print(f"{pass_type.upper()} OI: {oi:.3f}, {pass_type.upper()} Perf: {perf:.3e} FLOP/s")