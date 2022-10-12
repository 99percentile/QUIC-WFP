#!/usr/bin/env python
import os
import pandas as pd

directories = ['../hp131_cw/closed-world/', '../hp131_ow/open-world/']

for world in directories:
    for directory in os.listdir(world):
        if os.path.isdir(world + directory):
            print('DIR: ', world + directory)
            for filename in os.listdir(world + directory):
                if filename.endswith('.csv'):
                    print(filename)
                    while True:
                        try:
                            file = world+directory+'/'+filename
                            df = pd.read_csv(file, on_bad_lines='error', header=0)
                            break
                        except Exception as e:
                            line = int(str(e).split(' ')[-3][:-1])-1
                            print(line)
                            a_file = open(file, "r", encoding="utf-8")
                            list_of_lines = a_file.readlines()
                            print(list_of_lines[line])
                            list_of_lines[line] = ','.join(list_of_lines[line].split(',')[:11])+"\"\n"
                            if list_of_lines[line].endswith("\"\"\n"):
                                print('here')
                                list_of_lines[line] = list_of_lines[line][:-3] + "\"\n"
                            print(list_of_lines[line])
                            
                            a_file = open(file, "w", encoding="utf-8")
                            a_file.writelines(list_of_lines)
                            a_file.close()
