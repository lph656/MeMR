export CUDA_VISIBLE_DEVICES=0

bs=1
dropout=0.1
gradient_accumulation_steps=8
lr=1e-4
epoch=4

sql=512
tsql=64

A=neike
B=waike
C=erke
D=fuchanke
E=nanke
F=zhongliuke

# 指定持续学习中任务的顺序
task_list=${A}_${B}_${C}_${D}_${E}_${F}

META_EMBEDDINGS_PATH="./metadata_embeddings/keshi_meta_embeddings.pt"

python3 src/run_continual_causal_llama2.py \
    --model_name_or_path chinese-alpaca-plus-7b-hf \
    --task_list $task_list \
    --continual_learning \
    --mpeft_enabled \
    --matching_loss_v2 \
    --meta_embeddings_path $META_EMBEDDINGS_PATH \
    --do_train \
    --padding_strategy longest \
    --max_seq_length $sql \
    --max_target_length $tsql \
    --per_device_train_batch_size $bs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --output_dir checkpoints_continual_keshi_llama/order1_compose_peft \
    --overwrite_output_dir \
    --seed 0 \
    --save_strategy no \
    --evaluation_strategy no \
    --validation_split_percentage 0.1 \
    --overwrite_cache True \
    --lamda_1 0.05 \
    --lamda_2 0.01 \
    --orthogonal_threshold 0.2
