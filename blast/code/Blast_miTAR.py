# -*- coding: utf-8 -*-
# package for running the tool
#importing Packages

import os
from Blast_predicted_mirna_seq import Blast_seq # python wapper for ViennaRNA Package. 
           #"Please look into the documentation on its website for its installation".

data_path = 'data/mitar/transformer_mitar_miRNA.fasta'

miRNAs = []
output = []
cnt = 0
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n') # 구분자를 엔터로 나눈거 한 라인들!

with open('data/mitar/transformer_mitar_miRNA_IDs.fasta', 'r', encoding='utf-8') as k:
    miRNA_IDs = k.read().split('\n') # 구분자를 엔터로 나눈거 한 라인들!



for i in range(3000):
    # 정답
    miRNA_ID = miRNA_IDs[i].rstrip('\n') # 구분자 \t로 나눠줌
    # 예측한 시퀀스
    miRNA = lines[i].rstrip('\n')
    #miRNA_IDs.append(miRNA_ID)
    miRNAs.append(miRNA)

    after_blast = Blast_seq(miRNA)
    # print("-")
    # print("decoded:", miRNA)
    # print("miRNA_ID:", miRNA_ID)
    # print("after blast:", after_blast)
    if after_blast is None:
        continue
    if miRNA_ID in after_blast:
        cnt = cnt + 1

print(cnt)
    

