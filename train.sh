echo ============Job Starts!====================
model_name="microsoft/deberta-v3-base"
dataset_names=("rotten_tomatoes" "ag_news" "dair-ai/emotion")
output_dir_list=("./output/rotten_tomatoes_500" "./output/ag_news_500" "./output/dair-ai-emotion_500")
for data_idx in  {0..3}; do
       python ./train.py --learning_rate 2e-5 --num_train_epochs 6 --per_device_train_batch_size 24 \
       --model_name ${model_name}  \
       --output_dir ${output_dir_list[$data_idx]} \
       --dataset_names ${dataset_names[$data_idx]} \
       --dbg 0
done
