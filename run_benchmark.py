import os
import json
import argparse
import pandas as pd
from adapters.aipparel_adapter import AIpparelAdapter
# from adapters.sewformer_adapter import SewformerAdapter
from metrics.geometry import compute_geometry_metrics
# from metrics.topology import compute_topology_metrics
# from metrics.manufacturing import compute_manufacturing_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["aipparel", "sewformer"])
    parser.add_argument("--pred_dir", type=str, required=True, help="Raw outputs from the model")
    parser.add_argument("--gt_dir", type=str, required=True, help="Standard Spec JSON GTs")
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    # 1. 选择适配器
    if args.method == "aipparel":
        adapter = AIpparelAdapter()
    # elif args.method == "sewformer":
    #     adapter = SewformerAdapter()
    
    results = []
    
    # 2. 遍历测试集
    gt_files = [f for f in os.listdir(args.gt_dir) if f.endswith("_spec.json")]
    print(f"Start benchmarking {args.method} on {len(gt_files)} samples...")
    
    for gt_file in gt_files:
        sample_id = gt_file.replace("_spec.json", "")
        
        # 寻找对应的预测文件 (假设命名规则一致)
        # AIpparel 可能输出 .py 或 .txt
        pred_path = os.path.join(args.pred_dir, f"{sample_id}.py") 
        if not os.path.exists(pred_path):
            print(f"[Warn] Missing prediction for {sample_id}")
            continue
            
        gt_path = os.path.join(args.gt_dir, gt_file)
        with open(gt_path) as f: gt_spec = json.load(f)
        
        # 3. 适配转换
        try:
            pred_spec = adapter.convert(pred_path, gt_spec)
        except Exception as e:
            print(f"[Error] Adapter failed for {sample_id}: {e}")
            continue
            
        # 4. 计算指标
        geo_metrics = compute_geometry_metrics(pred_spec, gt_spec)
        # topo_metrics = compute_topology_metrics(pred_spec, gt_spec) # 需实现匹配算法
        # mfg_metrics = compute_manufacturing_metrics(pred_spec, gt_spec)
        
        # 5. 记录
        row = {"id": sample_id}
        row.update(geo_metrics)
        # row.update(topo_metrics)
        # row.update(mfg_metrics)
        results.append(row)
        
    # 6. 保存报告
    df = pd.DataFrame(results)
    print("\nBenchmark Summary:")
    print(df.mean(numeric_only=True))
    df.to_csv(args.output, index=False)
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()