export TORCH_DISTRIBUTED_DEBUG="DETAIL"

cd ..

data_path=ft-training_set/commonsense_170k.json
# data_path=ft-training_set/MetaMathQA.json
output_dir=trained_models/llama2-7b-commonsense
# base_model=meta-llama/Llama-2-7b-hf
base_model=Qwen/Qwen3-8B

WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=7860 finetune.py \
    --base_model $base_model \
    --data_path $data_path \
    --output_dir $output_dir \     --batch_size 16 \    --micro_batch_size 2 \    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \   --val_set_size 0 \
    --eval_step 80 \
    --save_step 80 \
    --adapter_name peanut \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 \
    --lora_alpha 32 \
    --load_8bit=False \
    --lora_dropout=0.05 \
    --WORLD_SIZE=4 \
    --train_on_inputs=False \