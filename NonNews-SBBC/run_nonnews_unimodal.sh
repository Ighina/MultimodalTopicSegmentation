bs=$1
expdir=$2
metric=$3

declare -a encoders=(NonNewsSentence/x-vectors NonNewsSentence/openl3/_mean_std NonNewsSentence/non_news_roberta NonNewsSentence/non_news_topseg NonNewsSentence/x-vectors+NonNewsSentence/openl3/_mean_std NonNewsSentence/non_news_roberta+NonNewsSentence/non_news_topseg)

for encoder in "${encoders[@]}"
    do
		expname=${expdir}/BiLSTM_bs${bs}_${encoder}
		python ../train_fit.py -exp ${expname} -s_last -arc BiLSTM -enc ${encoder} -lr 1e-3 -hs -huss 256 -nlss 2 -diss 0 0.2 0.5 -doss 0 0.2 0.5 -data NonNews -bs ${bs} -ef NonNewsSentence/${encoder} -lf NonNewsSentence/labs_dict.pkl --metric ${metric} -max 1000 -vp 0.15 -pat 50 -ar -as -split NonNews_split.json -loss FocalLoss
    done