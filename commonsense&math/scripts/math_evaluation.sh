export TORCH_DISTRIBUTED_DEBUG="DETAIL"

base_model=/hpc_stor03/sjtu_home/haoxiang.jiang/models/llama-2-hf
lora_weights=/hpc_stor03/sjtu_home/haoxiang.jiang/NEAT/trained_models/llama2-7b-NEAT-metamath

CUDA_VISIBLE_DEVICES=0 python /hpc_stor03/sjtu_home/haoxiang.jiang/NEAT/evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset MATH \
    --base_model $base_model \
    --lora_weights $lora_weights \