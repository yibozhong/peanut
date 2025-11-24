export TORCH_DISTRIBUTED_DEBUG="DETAIL"

cd ..

base_model=meta-llama/Llama-2-7b-hf
lora_weights=trained_models/llama2-7b-commonsense

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset piqa \
    --batch_size 8 \
    --base_model $base_model \
    --lora_weights $lora_weights \