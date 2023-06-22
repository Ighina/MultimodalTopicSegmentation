bs=$1
expdir=$2
metric=$3

declare -a encoders=(RadioNewsSentence/x-vectors RadioNewsSentence/openl3/_mean_std RadioNewsSentence/x-vectors+RadioNewsSentence/openl3/_mean_std)
declare -a encoders2=(RadioNewsSentence/radio_news_roberta RadioNewsSentence/radio_news_topseg RadioNewsSentence/radio_news_roberta+RadioNewsSentence/radio_news_topseg)

counter=0
for encoder in "${encoders[@]}"
    do
	counter2=0
    for encoder2 in "${encoders2[@]}"
        do
            expname=${expdir}/BiLSTM_bs${bs}_${encoder}+${encoder2}
            python ../train_fit.py -exp ${expname} -s_last -arc BiLSTMLateFusion -enc ${encoder} -enc2 ${encoder2} -lr 1e-3 -hs -huss 256 -nlss 2 -diss 0 0.2 0.5 -doss 0 0.2 0.5 -data RadioNews -bs ${bs} -ef ${encoder} -ef2 ${encoder2} -lf RadioNewsSentence/labs_dict.pkl --metric ${metric} -max 1000 -vp 0.15 -pat 50 -ar -as -split RadioNews_split.json -loss FocalLoss
        done
	counter+=1
    done