#evaluation
GeneSymbol = 'ABCC1'

cnt = 0
miDerma_data_path = 'data/Gene/DB_'+GeneSymbol+'.csv'
mirna_prediction_path = 'ABCC2.txt'

# miDerma DB에 있는 line빼오기
with open(miDerma_data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n') # 구분자를 엔터로 나눈거 한 라인들!
outofDB = []


for line in lines[1:]: # 일단 첫번째줄 빼고 가져오기
    Disease_Name,DiseaseID,Gene__Symbol,Gene_ID,Omim_ID,Dis_To_genePMID,MiRNA_name,miRTarBaseID = line.split(',') # 구분자 ,로 나눠줌
    print(MiRNA_name[1:-1])
    if MiRNA_name[1:-1] in output:
        print("in db miRNA is", MiRNA_name[1:-1])
        cnt = cnt+1
    else:
        outofDB.append(MiRNA_name[1:-1])


print('cnt!!!:',cnt)
print(outofDB)