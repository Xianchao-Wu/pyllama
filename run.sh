#########################################################################
# File Name: run.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Apr 18 22:44:52 2023
#########################################################################
#!/bin/bash

python -m ipdb inference.py \
	--ckpt_dir ./pyllama_data/7B/ \
	--tokenizer_path ./pyllama_data/tokenizer.model

