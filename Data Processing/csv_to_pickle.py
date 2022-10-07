import os
import pandas as pd

errors = []
directories = ['../hp131_cw/closed-world', '../hp131_ow/open-world']

for world in directories:
    for directory in os.listdir(world):
        if os.path.isdir(directory):
            print('DIR: ', directory)
            for filename in os.listdir(directory):
                if filename.endswith('.csv'):
                    print(filename)
                    try:
                        df = pd.read_csv(directory+'/'+filename)
                        df.to_pickle(directory+'/'+filename[:-8]+'.pickle')
                    except Exception as e:
                        print(directory+'/'+filename)
                        print("ERROR : "+str(e))
                        errors.append(directory+'/'+filename + str(e))
                        continue

