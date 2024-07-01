set -e
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
accelerate launch ./benchmark.py