bs=$1
expdir=$2
metric=$3

declare -a encoders=(x-vectors openl3/_mean_std radio_news_roberta radio_news_topseg)
counter=0
for encoder in "${encoders[@]}"
    do
	counter2=0
    for encoder2 in "${encoders[@]}"
        do
        if [[ ${counter2} -gt ${counter} ]];
            then
                expname=${expdir}/BiLSTM_bs${bs}_${encoder}+${encoder2}
                    python ../train_fit.py -exp ${expname} -s_last -arc BiLSTM -enc ${encoder}+${encoder2} -lr 1e-3 -hs -huss 256 -nlss 2 -diss 0 0.2 0.5 -doss 0 0.2 0.5 -data RadioNews -bs ${bs} -ef RadioNewsSentence/${encoder}+RadioNewsSentence/${encoder2} -lf RadioNewsSentence/labs_dict.pkl --metric ${metric} -max 1000 -vp 0.15 -pat 50 -ar -as -split RadioNews_split.json -loss FocalLoss
            fi
		counter2+=1
        done
	counter+=1
    done