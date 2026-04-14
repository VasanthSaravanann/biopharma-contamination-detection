#!/usr/bin/env python3
"""
Quick Start Script for Biopharmaceutical Contamination Detection System

This script provides a simplified entry point for running the pipeline.
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Biopharmaceutical Contamination Detection - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                    # Run quick demo (5-10 min)
  %(prog)s --full                    # Run full pipeline (30-60 min)
  %(prog)s --notebook                # Print notebook launch command
  %(prog)s --test                    # Run tests
        """
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demo pipeline with reduced data'
    )
    
    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Run full pipeline with default settings'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--notebook', '-n',
        action='store_true',
        help='Show command to launch Jupyter notebook'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run tests'
    )
    
    args = parser.parse_args()
    
    if args.notebook:
        print("\nTo launch the Jupyter notebook, run:")
        print("  jupyter notebook notebooks/experimentation.ipynb\n")
        return 0
    
    if args.test:
        print("\nRunning tests...")
        import pytest
        sys.exit(pytest.main(['tests/', '-v']))
    
    if args.demo or args.full:
        from run_pipeline import main as run_main
        
        # Build arguments
        cmd_args = ['--output', args.output]
        if args.demo:
            cmd_args.append('--demo')
        
        # Modify sys.argv for the pipeline
        original_argv = sys.argv
        sys.argv = ['run_pipeline.py'] + cmd_args
        
        try:
            run_main()
        finally:
            sys.argv = original_argv
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Start Guide:")
        print("="*60)
        print("\n1. Run demo pipeline:")
        print("   python quickstart.py --demo")
        print("\n2. Launch Jupyter notebook:")
        print("   python quickstart.py --notebook")
        print("\n3. Run tests:")
        print("   python quickstart.py --test")
        print("\n" + "="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
