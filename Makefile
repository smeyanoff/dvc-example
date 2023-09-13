prepare_stage1:
	poetry run python src/data/data_download.py

prepare_stage2:
	poetry run python src/data/data_prepare.py

data_prepare: prepare_stage1 prepare_stage2
