import sys, time, json, argparse
import numpy as np
import pandas as pd
from argparse import Namespace
from run_pipeline import ContaminationDetectionPipeline

class SmokePipeline(ContaminationDetectionPipeline):
    """Quick smoke test for pipeline integration."""
    pass

if __name__ == "__main__":
    # Create args object with required attributes
    args = Namespace(
        config="config/pipeline_config.yaml",
        experiment="EColi_10CFU",
        output="output"
    )
    
    # Run smoke test
    print("🔥 Starting smoke test (quick validation)...", flush=True)
    pipeline = ContaminationDetectionPipeline({}, "output")
    try:
        pipeline.run_fused_pipeline(args)
        print("✅ Smoke test PASSED", flush=True)
    except Exception as e:
        print(f"❌ Smoke test FAILED: {e}", flush=True)
        sys.exit(1)
