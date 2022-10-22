export PYTHONWARNINGS="ignore:semaphore_tracker:UserWarning"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u main.py --task_name "docred-t5-large-test" \
	--model_type "t5" \
	--pretrained_model_name_or_path "t5-large" \
	--data_dir "data/DocRED" \
	--train_data_type "train_annotated" \
	--load_model_path "checkpoint/modelS.pt1_129142" \
	--num_epochs 10 \
	--lr 5e-5 \
	--weight_decay 1e-2 \
	--do_test \
	--refresh \
  > docred-t5-large-test.log 2>&1 &