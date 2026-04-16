import sys, time, json
import numpy as np
import pandas as pd
from run_pipeline import ContaminationDetectionPipeline

class SmokePipeline(ContaminationDetectionPipeline):
    def run_fused_pipeline(self, args):
        # Force a small subset of the audit-audited data
        parser = AmbrDatasetParser("FCIC_AMBR_05/Data")
        ambr_df = parser.parse_sensor_files("00001/S").iloc[:150]
        fuser = DataFuser("FCIC_AMBR_05/Data", "Bacteria Contamination Work")
        fused_df = fuser.fuse(ambr_df, "EColi", "10CFU")
        # Logic...
        super().run_fused_pipeline(args)

c = ContaminationDetectionPipeline({}, 'output')
c.run_fused_pipeline(None)
