import numpy as np

class Q1_Configs:
    SAVE_DIR = "./outputs/figures"
    L1_MIN = 0
    L1_MAX = 100

    L2_MIN = 0
    L2_MAX = 10

    N_LAMBDAS = 25

    LEARNING_RATE = 0.05
    MAX_ITERS = 2000

    SEED = 42

class Q2_Configs:
    SAVE_DIR = "./outputs/figures"
    RESULTS_DIR = "./outputs/results"
    TEST_RATIO = 0.2

    COPYED_FEATURES = [2, 2, 2, 8, 8, 8]
    NOISE_STD = 0.01

    LAMBDA1 = 1
    LAMBDA2 = 1

    LEARNING_RATE = 0.05
    MAX_ITERS = 2000

    N_CLIP_EPS = 50
    CLIP_EPS_SHOW_L1 = 4.0
    CLIP_EPS_SHOW_L2 = 4.0

    SEED = 42

class Q3_Configs:
    SAVE_DIR = "./outputs/figures"
    TEST_RATIO = 0.2

    LEARNING_RATE = 0.05
    MAX_ITERS = 2000

    LAMBDA1 = 25.0
    LAMBDA2 = 1.0

    L1_MIN = 0.01
    L1_MAX = 10
    L2_MIN = 0.01
    L2_MAX = 1

    N_LAMBDAS = 25

class Extra_Configs:
    SQL_PATH = "data/extra_world_data/erdai_api_weather_static_heweather.sql"
    ORIG_DIR = "data/world_base/origin"   # 原始 train/valid: 提供国名词表并做地名去重
    OUT_DIR = "data/world_extra/origin"
    RESULTS_DIR = "./outputs/results"

    MAX_TOTAL = 8000
    MAX_PER_COUNTRY = 50
    MAX_US = 300
    HEAD_RATIO = 0.6

    SEED = 42

class Extract_Configs:
    MODEL_DIR = "Qwen3-4B-Base"
    BATCH_SIZE = 8

class Q4_Configs:
    RESULTS_DIR = "./outputs/results"
    SAVE_DIR = "./outputs/figures"

    N_LAYERS = 36
    # 与 Q5 同口径: 原始特征空间固定 λ2(取 Q5 CV 终选值,层排序对该值不敏感)
    LAMBDA2 = 4.33e-3

    SEED = 42

class Q5_Configs:
    SAVE_DIR = "./outputs/figures"
    RESULTS_DIR = "./outputs/results"

    N_FOLDS = 5
    N_LAMBDAS = 12
    # WithNorm = False(原始特征,σ² ≈ 0.05),网格较归一化版整体左移约 1/σ²
    L1_MIN = 1e-4
    L1_MAX = 1e-1
    L2_MIN = 1e-4
    L2_MAX = 1e2

    LASSO_MAX_ITERS = 5000

    SEED = 42

class Q6_Configs:
    SAVE_DIR = "./outputs/figures"
    RESULTS_DIR = "./outputs/results"

    MAP_DIR = "data/maps"

    SEED = 42

class Q7_Configs:
    SAVE_DIR = "./outputs/figures"
    RESULTS_DIR = "./outputs/results"

    MIN_VALID_PER_COUNTRY = 2

    SEED = 42

class Q8_Configs:
    SAVE_DIR = "./outputs/figures"
    RESULTS_DIR = "./outputs/results"

    TRUTH_DIR = "data/truth"
    LAYER = 19        # 沿用 Q9 最优层
    N_FOLDS = 5
    # 与 Q9 同网格: 标准化空间 C = 1/(2·λ2·n)
    C_MIN = 1e-4
    C_MAX = 1e1
    N_CS = 7
    TEST_RATIO = 0.2  # 域验证留出比例

    SEED = 42

class Q9_Configs:
    SAVE_DIR = "./outputs/figures"
    RESULTS_DIR = "./outputs/results"

    TRUTH_DIR = "data/truth"
    N_LAYERS = 36
    N_FOLDS = 5
    # 标准化空间 C = 1/(2·λ2·n)
    C_MIN = 1e-4
    C_MAX = 1e1
    N_CS = 7

    SEED = 42
