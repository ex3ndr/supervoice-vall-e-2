set -e
while true; do
    accelerate launch ./train.py || true
done