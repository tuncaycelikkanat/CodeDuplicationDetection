import os
import sys
import json
import argparse
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import Log, Colors
from utils.test_automation import run_automation, get_experiment_path

def main():
    parser = argparse.ArgumentParser(description="Run test automation across multiple models and scenarios, then save a consolidated JSON.")
    parser.add_argument("--exp-ids", nargs='+', type=int, required=True, help="List of experiment IDs to test (e.g., 76 77 78)")
    parser.add_argument("--out", type=str, default="evaluation/model_comparison.json", help="Output JSON path")
    args = parser.parse_args()

    scenarios = ["original", "imbalanced", "balanced"]
    
    # Big dictionary to hold all results
    comparison_data = {}

    for exp_id in args.exp_ids:
        exp_path = get_experiment_path(exp_id)
        if not exp_path:
            Log.warning(f"Experiment ID {exp_id} could not be found, skipping.")
            continue
            
        model_name = os.path.basename(exp_path)
        Log.step(f"Starting evaluations for Model: {model_name} (ID: {exp_id})")
        
        comparison_data[model_name] = {}
        
        for sc in scenarios:
            Log.substep(f"Running scenario: {sc.upper()}")
            test_dir = f"evaluation/test_clones_{sc}"
            
            # Catching the output of run_automation
            try:
                report, out_dir = run_automation(
                    test_dir=test_dir, 
                    threshold=0.95, # auto_thresh will overwrite this anyway
                    exp_id=exp_id, 
                    auto_thresh=True
                )
                
                # To keep the JSON file size manageable, we can remove the huge 'details' arrays
                # which have every single pair listed, but keep the global_y_true and per_type y_true arrays.
                # Actually, the user said "sana lazim olacak her seyi dahil ederek", so we will keep the arrays 
                # for plotting PR curves, but strip out the string codes to save space.
                
                if "per_type" in report:
                    for t_name, t_data in report["per_type"].items():
                        if "details" in t_data:
                            del t_data["details"] # Delete massive dicts, we only need y_true and y_prob for plotting
                
                comparison_data[model_name][sc] = report
                Log.success(f"Scenario {sc.upper()} for {model_name} completed.")
                
            except Exception as e:
                Log.error(f"Failed on {model_name} - {sc}: {str(e)}")
                continue

    # Save to disk
    out_path = os.path.join(_PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(comparison_data, f, indent=4)
        
    Log.success(f"All done! Combined results saved to {out_path}")

if __name__ == "__main__":
    main()
