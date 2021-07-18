import sys
import pandas as pd
import numpy as np


def read_fasta(fasta_file):
    try:
        fp = open(fasta_file)
    except IOError:
        print('cannot open '+fasta_file + ', check if it exist!')
        exit()
    else:
        fp = open(fasta_file)
        lines = fp.readlines()
        
        fasta_dict = {} #record seq for one id
        idlist=[] #record id list sorted
        gene_id = ""
        for line in lines:
            line = line.replace('\r','')
            if line[0] == '>':
                if gene_id != "":
                    fasta_dict[gene_id] = seq.upper()
                    idlist.append(gene_id)
                seq = ""
                gene_id = line.strip('\n') #  line.split('|')[1] all in > need to be id
            else:
                seq += line.strip('\n')
        
        fasta_dict[gene_id] = seq.upper() #last seq need to be record
        idlist.append(gene_id)

    return fasta_dict,idlist


def get_sequence_odd_fixed(fasta_dict,idlist, window=20, label=1):
    seq_list_2d = []
    id_list = []
    pos_list = []
    for id in idlist: #for sort
        seq = fasta_dict[id]
        final_seq_list = [label] + [ AA for AA in seq] 

        id_list.append(id)
        pos_list.append(window)
        seq_list_2d.append(final_seq_list)
        
    df = pd.DataFrame(seq_list_2d)
    df2= pd.DataFrame(id_list)
    df3= pd.DataFrame(pos_list)
    
    return df,df2,df3


def analyseFixedPredict(fasta_file, window=20, label=1):
    fasta_dict,idlist = read_fasta(fasta_file) 

    sequence,ids,poses= get_sequence_odd_fixed(fasta_dict,idlist, window, label)
    
    return sequence,ids,poses