# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
from Bio.Blast.Applications import NcbiblastnCommandline

#mirna = '>'+'refseq_1'+'\n'+'UCACCAGCCCUGUGUUCCCUAG'
def Blast_seq(mirna):

    with open('data/mirna.fasta','w+') as f:
        f.write('>'+'refseq_1'+'\n'+ str(mirna))
    if os.path.isfile('blast_result.csv'):   
        os.remove('blast_result.csv')    
    blastx_cline = NcbiblastnCommandline(query='data/mirna.fasta', db="/home/newuser/ML/blast/data/blast_db/miRBase2.fasta", evalue=0.01,outfmt=10, out="blast_result.csv",word_size= 7, gapopen = 50, gapextend = 3, strand= 'both')
    stdout, stderr = blastx_cline()
    list_of_mirna = []
    try: 
        with open('blast_result.csv','r+') as f:
            lines = f.read()
            if '\n' in lines:
                lines = lines.split('\n')
            
            for line in lines:
                if ',' in line:
                    list_of_mirna.append(line.split(',')[1])
        if len(list_of_mirna)>0:
            
            return list_of_mirna        
        else:
            return None
    except:
        return None