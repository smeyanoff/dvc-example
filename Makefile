prepare_dirs:
	mkdir -p data/initial_data
	mkdir data/prepared_data

prepare_stage1:
	poetry run python src/data/data_download.py

prepare_stage2:
	poetry run python src/data/data_prepare.py

data_prepare: prepare_dirs prepare_stage1 prepare_stage2

training:
	poetry run python src/models/model_train.py
