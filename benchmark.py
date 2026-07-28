"""
benchmark.py
============
Comprehensive performance benchmarking for thermal face recognition model.

Measures:
  • Inference latency (ms per image)
  • Model size (parameters, file size)
  • Memory usage (peak, average)
  • Throughput (images/second)
  • Hardware utilization

Usage:
    python benchmark.py --checkpoint_dir checkpoints --data_dir data
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Import model class
import sys
sys.path.insert(0, os.path.dirname(__file__))

class DualHeadFaceNet(nn.Module):
    def __init__(self, num_persons: int, num_expressions: int, dropout: float = 0.4):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.identity_head = nn.Linear(512, num_persons)
        self.expression_head = nn.Linear(512, num_expressions)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        feat = self.shared_fc(x)
        return self.identity_head(feat), self.expression_head(feat)


class Benchmark:
    def __init__(self, checkpoint_dir: str = "checkpoints", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        
        # Load checkpoint
        ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        self.model = DualHeadFaceNet(
            num_persons=ckpt.get("num_persons", 113),
            num_expressions=ckpt.get("num_expressions", 5),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        
        self.results = {}

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def get_model_size(self):
        """Get model file size in MB."""
        ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
        return size_mb

    def measure_latency(self, num_warmup=10, num_trials=100, batch_size=1):
        """Measure inference latency."""
        dummy_input = torch.randn(batch_size, 3, 128, 128, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(dummy_input)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_trials):
                t0 = time.time()
                _ = self.model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
        
        times = np.array(times[10:])  # skip first few
        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
            "batch_size": batch_size,
        }

    def measure_memory(self):
        """Measure peak GPU/CPU memory usage."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        dummy_input = torch.randn(1, 3, 128, 128, device=self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        if self.device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return {"peak_memory_mb": peak_memory_mb}
        else:
            return {"peak_memory_mb": "N/A (CPU)"}

    def compute_flops(self):
        """Estimate FLOPs (floating point operations)."""
        try:
            from thop import profile
            dummy_input = torch.randn(1, 3, 128, 128, device=self.device)
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            return flops / 1e6, params / 1e6  # Return in millions
        except ImportError:
            return None, None

    def run_all(self):
        """Run all benchmarks."""
        print("\n" + "="*70)
        print("  THERMAL FACE RECOGNITION – BENCHMARK REPORT")
        print("="*70)
        
        # Model size
        total_params, trainable = self.count_parameters()
        file_size_mb = self.get_model_size()
        print(f"\n── MODEL SIZE ──")
        print(f"  Total Parameters       : {total_params:,}")
        print(f"  Trainable Parameters   : {trainable:,}")
        print(f"  Model File Size        : {file_size_mb:.2f} MB")
        
        self.results["model_size"] = {
            "total_params": total_params,
            "trainable_params": trainable,
            "file_size_mb": file_size_mb,
        }
        
        # FLOPs
        print(f"\n── COMPUTATIONAL COMPLEXITY ──")
        mflops, _ = self.compute_flops()
        if mflops:
            print(f"  FLOPs per Inference    : {mflops:.2f}M FLOPs")
            self.results["flops"] = mflops
        else:
            print(f"  FLOPs per Inference    : Requires 'thop' library")
        
        # Latency (batch 1)
        print(f"\n── INFERENCE LATENCY (Batch=1) ──")
        lat_b1 = self.measure_latency(batch_size=1, num_trials=100)
        print(f"  Mean               : {lat_b1['mean_ms']:.2f} ms")
        print(f"  Std Dev            : {lat_b1['std_ms']:.2f} ms")
        print(f"  Min / Max          : {lat_b1['min_ms']:.2f} / {lat_b1['max_ms']:.2f} ms")
        print(f"  Throughput (1 img) : {1000 / lat_b1['mean_ms']:.1f} images/sec")
        
        self.results["latency_batch1"] = lat_b1
        
        # Latency (batch 32)
        print(f"\n── INFERENCE LATENCY (Batch=32) ──")
        lat_b32 = self.measure_latency(batch_size=32, num_trials=50)
        print(f"  Mean (per batch)   : {lat_b32['mean_ms']:.2f} ms")
        print(f"  Mean (per image)   : {lat_b32['mean_ms'] / 32:.2f} ms")
        print(f"  Throughput         : {32 * 1000 / lat_b32['mean_ms']:.1f} images/sec")
        
        self.results["latency_batch32"] = lat_b32
        
        # Memory
        print(f"\n── MEMORY USAGE ──")
        mem = self.measure_memory()
        for k, v in mem.items():
            if isinstance(v, float):
                print(f"  {k.replace('_', ' ').title()} : {v:.2f} MB")
            else:
                print(f"  {k.replace('_', ' ').title()} : {v}")
        
        self.results["memory"] = mem
        
        # Device info
        print(f"\n── HARDWARE ──")
        print(f"  Device             : {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU                : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Capability    : {torch.cuda.get_device_capability(0)}")
        
        self.results["device"] = str(self.device)
        
        print("\n" + "="*70 + "\n")
        
        return self.results

    def save_report(self, output_path: str = "benchmark_report.json"):
        """Save benchmark results as JSON."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  Benchmark report saved → {output_path}")


def main(args):
    device = torch.device(args.device)
    benchmark = Benchmark(args.checkpoint_dir, device=device)
    benchmark.run_all()
    benchmark.save_report(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermal Face Recognition Benchmark")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Directory containing best_model.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to benchmark on")
    parser.add_argument("--output", default="benchmark_report.json",
                        help="Output JSON file for results")
    args = parser.parse_args()
    main(args)
