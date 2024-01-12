echo ================================


model_name_list=("./output/ag_news_500/checkpoint-500")  
output_dir_list=("./output/test")
python ./train.py --learning_rate 2e-5 --num_train_epochs 6 --per_device_train_batch_size 24 \
        --model_name ${model_name_list[0]}  \
        --model_path ${model_name_list[0]} \
        --dataset_names ag_news \
        --output_dir ${output_dir_list[0]} \
        --dbg 1
done