echo ============Job Starts!====================
model_name=("./output/rotten_tomatoes_500" "./output/ag_news_500" "./output/dair-ai-emotion_500")
dataset_names=("rotten_tomatoes" "ag_news" "dair-ai/emotion")
output_dir_list=("./output/rt-rt" "./output/ag-ag" "./output/da-da")
for data_idx in  {0..3}; do
       echo ${model_name[$data_idx]}
       python ./train.py --learning_rate 2e-5 --num_train_epochs 6 --per_device_train_batch_size 24 \
       --model_name ${model_name[$data_idx]}  \
       --model_path ${model_name[$data_idx]} \
       --output_dir ${output_dir_list[$data_idx]} \
       --dataset_names ${dataset_names[$data_idx]} \
       --dbg 0
       
done
