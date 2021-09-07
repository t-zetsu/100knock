import pandas as pd

input_file = "outputs/64.csv"

def main():
    data = pd.read_csv(input_file, index_col=0, sep='\t')

    sem_cnt = 0
    sem_cor = 0
    syn_cnt = 0
    syn_cor = 0

    for row in data.itertuples():
        if row[1].startswith("gram"):
            syn_cnt += 1
            if row[5] == row[6]:
                syn_cor += 1
        else:
            sem_cnt += 1
            if row[5] == row[6]:
                sem_cor += 1

    print(f'意味的アナロジー正解率: {sem_cor/sem_cnt:.3f}')
    print(f'文法的アナロジー正解率: {syn_cor/syn_cnt:.3f}') 
            
main()