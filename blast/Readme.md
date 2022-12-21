# Seq2seq 논문에 대하여 mitar논문 데이터에 대한 정확성 평가 및 구현, Attention 추가

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
***
## 22/07/13
### miTAR에서 사용한 데이터로 Seq2seq모델에 적용해보기.  
 
||E-value = 0.01|E-value = 0.005|E-value = 0.001|
|------|:---:|:---:|:---:|
|miTAR|3.5%|1.2%|0.3%|
|seq2seq|100% |100% |100%|
***
## 22/07/20
### seq2seq data를 miTar 모델에 넣어서 돌려보기   
그런데 데이터의 모든 LABEL을 1로 하고 학습을 시켰더니 모든 정확성이 100%가 나오는지도 모르고 끝날때 쯤에 봐서 틀린 DATA도 넣어서 학습 중에 있습니다.   
오답 데이터가 필요합니다.   
-> 오답데이터를 일부러 섞어서 생성한 것으로 활용하겠습니다.   

***
## 22/07/27
### seq2seq data를 miTar 모델에 넣어서 돌려보기

데이터는 10000개는 정상적인 data를 넣고 label을 1로 했고,   
9000개는 틀린 data를 넣고 label을 0으로 해서 학습을 시켰습니다.   

batch= 100 lr= 0.005 dout= 0.4 acc= 90.18421173095703   
batch size가 100이고, learning rate 가 0.005이고, dropout 이 0.4 일때 **90%로 가장 높은 정확성을 보였습니다.  ** 

***
## 22/08/03
### miTAR 데이터가 일단 웹에서 Blast로 검색이 되는지   
: 나오지 않았습니다. 논문에서는 local data로 사용했다고 해서, 로컬 db에 넣는 방법을 찾고있습니다.   
 Then these predicted microRNA sequences will be mapped to their microRNA IDs using BLASTx algorithm on local database containing microRNA ID and respective sequences retrieved from mirBase Release 22, March 2018- hence giving a list of predicted microRNA IDs as output.   
 - 로컬 DB에 넣기 성공했습니다. 해당 파일은 Blast_AddData.py   
 
### seq2seq data를 miTar 모델에 넣어서 돌려보고 그 결과를 blast   
 miRNA를 Blast를 통해 돌려봤더니 10000개 중에 9997개가 다 blast가 되었습니다   
 decoded된 miRNA가 원래 miRNA 이름이랑 맞는지를 확인하면 10000개중에 6341개가 일치하였습니다. 63.41%의 정확성   

***
## 22/08/23
### seq2seq 모델에 Attention을 적용한다면 정확성에 도움이 되는지
구현은 트랜스포머(Transformer)로 하였습니다. 이는 2017년 구글이 발표한 논문인 "Attention is all you need"에서 나온 모델로 기존의 seq2seq의 구조인 인코더-디코더를 따르면서도, 논문의 이름처럼 어텐션(Attention)만으로 구현한 모델입니다.

![모델 구현](https://user-images.githubusercontent.com/101859033/208854428-ae917935-dce2-44e7-8a40-095660d6df22.png)
![결과](https://user-images.githubusercontent.com/101859033/208854511-b3f6e1fb-4346-46d9-9b51-798e06c1c15d.png)

***
## 22/08/31
### 기존 seq2seq 모델 학습 후 실행 해보기

![비교](https://user-images.githubusercontent.com/101859033/208860234-86dd1404-5f0b-4841-99c4-e2a5c570f47b.png)

![결과2](https://user-images.githubusercontent.com/101859033/208855133-10f65f1d-ca44-48d4-bf1e-3b9220cce332.png)

논문에서 말하는 정확성, 직접 코드를 보고 학습하고 평가한 정확성, Attention을 적용한 코드로 학습하고 평가한 정확성을 비교한 표

***
## 22/09/07

![결과3](https://user-images.githubusercontent.com/101859033/208855860-7d598f34-aa59-4888-b2b2-7f71ee234aa4.png)

miTAR 데이터와 seq2seq 데이터를 둘 다 Attention 모델에 학습을 해본 결과입니다.  
현저히 정확성이 떨어지는 결과가 나왔습니다.

문제를 다음과 같이 인풋 데이터에 대한 길이 차이로 봤습니다.
![문제](https://user-images.githubusercontent.com/101859033/208856234-cfdf8d6a-d418-4ab9-aab4-d68e9a05c6c7.png)

두 논문에서 사용하는 miRNA의 길이가 왜 다른지는 정확히 파악을 할 수 없었습니다.

