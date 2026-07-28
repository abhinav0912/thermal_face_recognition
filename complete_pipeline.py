"""
complete_pipeline.py
====================
One-command execution for entire thermal face recognition pipeline.

Runs in order:
  1. Data preparation
  2. Model training
  3. Evaluation & error analysis
  4. Visualizations
  5. Benchmarking
  6. Model export for deployment

Usage:
    python complete_pipeline.py --full  # Everything
    python complete_pipeline.py --quick # Skip ablation
    python complete_pipeline.py --analyze_only  # Skip training
"""

import argparse
import subprocess
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime


class PipelineRunner:
    def __init__(self, data_dir="data", checkpoint_dir="checkpoints", full_mode=True):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.full_mode = full_mode
        self.results = {}
        self.start_time = time.time()

    def run_command(self, cmd, description):
        """Run a command and capture output."""
        print(f"\n{'='*70}")
        print(f"  {description}")
        print(f"{'='*70}")
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=False)
            if result.returncode == 0:
                print(f"  ✓ {description} completed successfully")
                return True
            else:
                print(f"  ✗ {description} failed (exit code: {result.returncode})")
                return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def step_prepare_data(self):
        """Step 1: Data preparation"""
        cmd = f"python prepare_data.py --rgb_zip RGB-faces-128x128.zip --thermal_zip thermal-face-128x128.zip --out_dir {self.data_dir}"
        return self.run_command(cmd, "STEP 1: Data Preparation")

    def step_train_model(self):
        """Step 2: Model training"""
        cmd = f"python train.py --data_dir {self.data_dir} --output_dir {self.checkpoint_dir} --epochs 40 --batch_size 32"
        return self.run_command(cmd, "STEP 2: Model Training")

    def step_evaluate(self):
        """Step 3: Evaluation"""
        cmd = f"python evaluate.py --checkpoint_dir {self.checkpoint_dir} --data_dir {self.data_dir} --output_dir {self.checkpoint_dir}"
        return self.run_command(cmd, "STEP 3: Model Evaluation")

    def step_error_analysis(self):
        """Step 4: Error analysis"""
        cmd = f"python error_analysis.py --checkpoint_dir {self.checkpoint_dir} --data_dir {self.data_dir} --output error_analysis.json"
        return self.run_command(cmd, "STEP 4: Error Analysis")

    def step_visualizations(self):
        """Step 5: Generate visualizations"""
        cmd = f"python visualization.py --checkpoint_dir {self.checkpoint_dir} --data_dir {self.data_dir} --output_dir visualizations"
        return self.run_command(cmd, "STEP 5: Visualizations")

    def step_benchmark(self):
        """Step 6: Benchmark"""
        cmd = f"python benchmark.py --checkpoint_dir {self.checkpoint_dir} --output benchmark_report.json"
        return self.run_command(cmd, "STEP 6: Benchmarking")

    def step_ablation(self):
        """Step 7: Ablation study (optional, resource-intensive)"""
        if not self.full_mode:
            print(f"\n  [SKIPPED] Ablation study (use --full for full pipeline)")
            return True
        
        cmd = f"python ablation_study.py --data_dir {self.data_dir} --epochs 15 --mode quick --output ablation_study.json"
        return self.run_command(cmd, "STEP 7: Ablation Study")

    def step_export(self):
        """Step 8: Model export for deployment"""
        cmd = f"python model_export.py --checkpoint_dir {self.checkpoint_dir} --export_dir exported_models"
        return self.run_command(cmd, "STEP 8: Model Export")

    def generate_report(self):
        """Generate final summary report."""
        print(f"\n{'='*70}")
        print(f"  PIPELINE SUMMARY")
        print(f"{'='*70}\n")
        
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": int(elapsed),
            "total_time_readable": f"{hours}h {minutes}m {seconds}s",
            "mode": "full" if self.full_mode else "quick",
            "outputs": {
                "model": f"{self.checkpoint_dir}/best_model.pth",
                "training_curves": f"{self.checkpoint_dir}/training_curves.png",
                "error_analysis": "error_analysis.json",
                "visualizations": "visualizations/",
                "benchmark_report": "benchmark_report.json",
                "exported_models": "exported_models/",
            },
            "next_steps": [
                "1. Review visualizations: visualizations/*.png",
                "2. Check performance: cat benchmark_report.json",
                "3. Analyze errors: cat error_analysis.json",
                "4. Deploy model: Use exported_models/ for production",
                "5. Present results: Use generated visualizations in slides",
            ],
        }
        
        print(f"  Total Time:        {report['total_time_readable']}")
        print(f"  Mode:              {report['mode']}")
        print(f"\n  Output Files:")
        for name, path in report["outputs"].items():
            if isinstance(path, str):
                print(f"    • {name:.<30} {path}")
        
        print(f"\n  Next Steps:")
        for step in report["next_steps"]:
            print(f"    {step}")
        
        # Save report
        with open("pipeline_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n  Report saved → pipeline_report.json")
        print(f"\n{'='*70}\n")
        
        return report

    def run_full_pipeline(self):
        """Execute complete pipeline."""
        print("\n" + "="*70)
        print("  THERMAL FACE RECOGNITION – COMPLETE PIPELINE")
        print("="*70)
        print(f"\n  Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Mode: {'Full (with ablation)' if self.full_mode else 'Quick (skip ablation)'}")
        
        steps = [
            ("prepare", self.step_prepare_data, "Data Preparation"),
            ("train", self.step_train_model, "Training"),
            ("evaluate", self.step_evaluate, "Evaluation"),
            ("error_analysis", self.step_error_analysis, "Error Analysis"),
            ("visualizations", self.step_visualizations, "Visualizations"),
            ("benchmark", self.step_benchmark, "Benchmarking"),
            ("ablation", self.step_ablation, "Ablation Study"),
            ("export", self.step_export, "Export"),
        ]
        
        completed = []
        failed = []
        
        for step_name, step_func, description in steps:
            try:
                success = step_func()
                if success:
                    completed.append(step_name)
                else:
                    failed.append(step_name)
            except Exception as e:
                print(f"\n  ✗ Exception in {step_name}: {e}")
                failed.append(step_name)
        
        # Generate report
        self.generate_report()
        
        # Print summary
        print(f"\n  Completed Steps ({len(completed)}/{len(steps)}):")
        for step in completed:
            print(f"    ✓ {step}")
        
        if failed:
            print(f"\n  Failed Steps ({len(failed)}):")
            for step in failed:
                print(f"    ✗ {step}")
            return False
        
        return True


def main(args):
    runner = PipelineRunner(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        full_mode=args.full
    )
    
    if args.analyze_only:
        print("\n  [ANALYSIS ONLY MODE]")
        print("  Skipping training, running analysis on existing model...\n")
        runner.step_error_analysis()
        runner.step_visualizations()
        runner.step_benchmark()
        runner.generate_report()
    else:
        success = runner.run_full_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete Thermal Face Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python complete_pipeline.py --full           # Full pipeline with ablation
  python complete_pipeline.py                  # Quick pipeline (no ablation)
  python complete_pipeline.py --analyze_only   # Analysis only (skip training)
        """
    )
    parser.add_argument("--data_dir", default="data",
                       help="Data directory (default: data)")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                       help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--full", action="store_true",
                       help="Include ablation study (resource-intensive)")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only run analysis on existing model (skip training)")
    args = parser.parse_args()
    main(args)
