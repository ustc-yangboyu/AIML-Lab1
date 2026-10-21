"""从 CSV 抽取 Qwen3-4B-Base 中间表征"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.utils import extract_layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required = True, help = '输入 CSV(sentence 列)')
    parser.add_argument('--output-dir', required = True, help = '输出目录,存 {stem}.layer{L}.npy')
    parser.add_argument('--layers', nargs = '+', type = int, default = [11, 23, 35])
    parser.add_argument('--token-selection', default = 'country', choices = ['country', 'last'])
    args = parser.parse_args()
    extract_layers(args.input, args.output_dir, args.layers, args.token_selection)
