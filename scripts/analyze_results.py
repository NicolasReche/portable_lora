import os, json, glob

def analyze():
    print("================================================================")
    print("                     PIPELINE RESULTS                           ")
    print("================================================================")
    
    files = sorted(glob.glob("outputs/*.json") + glob.glob("results/*.json"))
    if not files:
        print("No result JSON files found in outputs/ or results/.")
        return
        
    for file in files:
        print(f"\n--- {os.path.basename(file)} ---")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'test_sets' in data:
                for set_name, set_data in data['test_sets'].items():
                    print(f"  Test Set: {set_name}")
                    eval_results = set_data.get('evaluation_results', {})
                    for metric, metric_res in eval_results.items():
                        if isinstance(metric_res, dict) and 'overall' in metric_res:
                            print(f"    - {metric}: {metric_res['overall']:.4f}")
                        else:
                            print(f"    - {metric}: {metric_res}")
            else:
                for key, val in data.items():
                    if isinstance(val, (int, float)):
                        print(f"  {key}: {val:.4f}")
                    elif isinstance(val, str):
                        print(f"  {key}: {val}")
        except Exception as e:
            print(f"  Error reading file: {e}")
            
if __name__ == '__main__':
    analyze()
