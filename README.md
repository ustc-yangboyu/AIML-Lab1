# AIML Lab1 — LLM 中间层特征的线性回归实验

## 仓库结构

```
├── configs.py                  # 各题超参数
├── scripts/
│   ├── extract_all_layers.py   # 从 CSV 抽取 Qwen3-4B 中间表征(GPU)
│   ├── build_extra_world.py    # 由第三方城市数据库构建 world_extra 数据集
│   └── Q1.py ... Q9.py         # 九道思考题实验
├── src/                        # 共享 datasets / models / utils
├── slurm/                      # 集群作业脚本(extract / cpu / run_all)
├── data/                       # 原始 CSV 与抽取特征(见「数据准备」)
├── outputs/
│   ├── figures/                # 实验图
│   └── results/                # 实验指标与最终权重(npz / json)
└── docs/Problems.pdf           # 实验题目
    └── report/                 # 实验报告(report.tex / report.pdf)
```

## 环境

Python 3.12,使用 [uv](https://docs.astral.sh/uv/) 管理依赖:

```bash
uv sync   # 创建 .venv 并按 uv.lock 安装依赖
```

## 数据准备

模型权重与抽取特征体积较大,不入库,需手动准备:

1. **模型权重**:下载 Qwen3-4B-Base 到 `Qwen3-4B-Base/`:

   ```bash
   modelscope download --model Qwen/Qwen3-4B-Base --local_dir Qwen3-4B-Base
   ```

2. **课程数据**:将下发的 `train.csv` / `valid.csv` 放入 `data/truth/origin/` 与 `data/world_base/origin/`。

3. **第三方城市数据库**:将全球城市库 SQL 放到
   `data/extra_world_data/erdai_api_weather_static_heweather.sql`(来源与许可说明见同目录 README,MIT)。

4. **构建 world_extra 数据集**:

   ```bash
   python scripts/build_extra_world.py
   ```

5. **抽取中间层特征**(GPU,全 36 层):

   ```bash
   # 以 world_base 为例;其余数据集同理
   python scripts/extract_all_layers.py \
       --input data/world_base/origin/train.csv \
       --output-dir data/world_base/processed \
       --layers 0 1 2 ... 35 --token-selection country
   ```

   集群上可直接 `sbatch slurm/extract.sbatch`。

## 运行实验

- **Q1–Q3**(CPU,相互独立):

  ```bash
  python scripts/Q1.py          # 或 Q2.py / Q3.py
  # 集群: sbatch --export=NAME=q1,SCRIPT=Q1.py slurm/cpu.sbatch
  ```

- **Q4–Q9**(有依赖,按序执行):

  ```bash
  sbatch slurm/run_all.sbatch   # 顺序 Q4 → Q5 → Q9 → Q8 → Q6 → Q7
  ```

  依赖关系:Q5 读 Q4 的最优层;Q6/Q7 读 Q5/Q9 的产物。

所有图表与指标输出到 `outputs/figures/` 与 `outputs/results/`;
实验报告在 `docs/report/` 

## 结果

最终各题结论与图表见 `outputs/report/report.pdf`。
