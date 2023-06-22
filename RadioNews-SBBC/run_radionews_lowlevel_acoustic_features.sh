bs=$1
expdir=$2
metric=$3

declare -a encoders=(RadioNewsSentence/radio_news_roberta RadioNewsSentence/radio_news_topseg RadioNewsSentence/radio_news_roberta+RadioNewsSentence/radio_news_topseg)

for encoder in "${encoders[@]}"
    do    
		expname=${expdir}/BiLSTM_bs${bs}_${encoder}
		python ../train_fit.py -exp ${expname} -s_last -arc BiLSTM -enc ${encoder} -lr 1e-3 -hs -huss 256 -nlss 2 -diss 0 0.2 0.5 -doss 0 0.2 0.5 -data RadioNews -bs ${bs} -ef ${encoder} -lf RadioNewsSentence/labs_dict.pkl --metric ${metric} -max 1000 -vp 0.15 -pat 50 -ar -as -split RadioNews_split.json -loss FocalLoss --timing_file nltk_sents_timings.pkl
        
    done