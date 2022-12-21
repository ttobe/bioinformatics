
GeneSymbol = 'ABCC1'

data_path = 'data/Gene/DB_.' + GeneSymbol+'txt'

miRNA_IDs = []
miRNAs = []
Accessions = []

output = []
cnt = 0
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n') # 구분자를 엔터로 나눈거 한 라인들!


for line in lines[1:]:
    # for mitar data
    DiseaseName,DiseaseID,GeneSymbol,GeneID,OmimID,DisTogenePMID,MiRNA,miRTarBaseID = line.split(',') # 구분자 \t로 나눠줌
    # for s2s data
    # miRNA,mRNA = line.split(',') # 구분자 \t로 나눠줌
    
    miRNA_IDs.append(miRNA_ID)
    
    Accessions.append(mRNA_Accession_Number)

    if miRNA not in miRNAs:
        miRNAs.append(miRNA)
        with open('data/blast_db/miRBase2.fasta','a+') as aa:
            aa.write('\n' + '>'+str(miRNA_ID)+' '+ str(mRNA_Accession_Number) + '\n' + str(miRNA) )

    