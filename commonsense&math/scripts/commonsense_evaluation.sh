export TORCH_DISTRIBUTED_DEBUG="DETAIL"

# base_model=/hpc_stor03/sjtu_home/haoxiang.jiang/models/llama-2-hf
base_model=/hpc_stor03/sjtu_home/haoxiang.jiang/models/llama3-8b
lora_weights=/hpc_stor03/sjtu_home/haoxiang.jiang/NEAT/trained_models/llama3-8b-xlora-commonsense

CUDA_VISIBLE_DEVICES=0 python /hpc_stor03/sjtu_home/haoxiang.jiang/NEAT/commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset piqa \
    --batch_size 8 \
    --base_model $base_model \
    --lora_weights $lora_weights \