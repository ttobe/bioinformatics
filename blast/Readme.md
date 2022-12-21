Seq2seq 논문에 대하여 mitar논문 데이터에 대한 정확성 평가

Seq2seq 논문: Predicting MicroRNA Sequence Using CNN and LSTM Stacked in Seq2Seq Architecture
https://ieeexplore.ieee.org/document/8807144

miTAR 논문: miTAR: a hybrid deep learning-based approach for predicting miRNA targets
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04026-6

두 논문의 차이는 방법의 차이
miTAR는 CNN과 BiRNN을 통해서 miRNA의 target 예측
input: miRNA+mRNA output: label(0 or 1)

Seq2seq는 Encoder-Decoder 학습을 통해 miRNA를 예측
모델 input: mRNA, output: miRNA
후 이 miRNA로 Blast를 통해 input mRNA_ID가 있는지 확인

22/7/13
1. miTAR에서 사용한 데이터로 Seq2seq모델에 적용해보기.
2. 
