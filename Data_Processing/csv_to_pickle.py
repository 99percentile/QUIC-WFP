import os
import pandas as pd

errors = []
directories = ['../hp131_cw/closed-world/', '../hp131_ow/open-world/']

for world in directories:
    for directory in os.listdir(world):
        if os.path.isdir(world+directory):
            print('DIR: ', directory)
            for filename in os.listdir(world+directory):
                if filename.endswith('.csv'):
                    filename = world + directory+'/'+filename
                    print(filename)
                    try:
                        df = pd.read_csv(filename)
                        df.to_pickle(filename[:-8]+'.pickle')
                    except Exception as e:
                        print(filename)
                        print("ERROR : "+str(e))
                        errors.append(filename + str(e))
                        continue

