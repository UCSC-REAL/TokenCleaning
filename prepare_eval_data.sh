
mkdir -p eval_data/downloads
mkdir -p eval_data/eval


mkdir -p eval_data/eval/tydiqa
wget -P eval_data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json
wget -P eval_data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json