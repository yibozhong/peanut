export TORCH_DISTRIBUTED_DEBUG="DETAIL"

data_path=/hpc_stor03/sjtu_home/haoxiang.jiang/NEAT/ft-training_set/commonsense_170k.json
# data_path=/hpc_stor03/sjtu_home/haoxiang.jiang/NEAT/ft-training_set/transformed_MetaMathQA.json
output_dir=/hpc_stor03/sjtu_home/haoxiang.jiang/NEAT/trained_models/llama2-7b-NEAT-commonsense-test
base_model=/hpc_stor03/sjtu_home/haoxiang.jiang/models/llama-2-hf
# base_model=/hpc_stor03/sjtu_home/haoxiang.jiang/models/llama3-8b

WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=7860 /hpc_stor03/sjtu_home/haoxiang.jiang/NEAT/finetune.py \
    --base_model $base_model \
    --data_path $data_path \
    --output_dir $output_dir \     --batch_size 16 \    --micro_batch_size 2 \    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \   --val_set_size 0 \
    --eval_step 80 \
    --save_step 80 \
    --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 \
    --lora_alpha 32 \
    --load_8bit=False \
    --lora_dropout=0.05 \
    --WORLD_SIZE=4 \
    --train_on_inputs=False \
    