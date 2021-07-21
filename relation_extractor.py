import os
import subprocess
import glob
import pandas as pd

def Stanford_Relation_Extractor():

    
    print('Relation Extraction Started')
    current_directory = os.getcwd()
    for f in glob.glob(os.getcwd() + "/data/output/kg/*.txt"):        
        print("Extracting relations for " + f.split("/")[-1])
        
        os.chdir(current_directory + '/stanford-openie')

        p = subprocess.Popen(['./process_large_corpus.sh',f,f + '-out.csv'], stdout=subprocess.PIPE)

        output, err = p.communicate()
   

    print('Relation Extraction Completed')


if __name__ == '__main__':
    Stanford_Relation_Extractor()
