#!/bin/bash

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com

pip install pytorch-lightning wandb ml_collections termcolor Biopython plotly
pip install tmtools DockQ dm-tree
pip install pdbfixer
