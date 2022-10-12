import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics
import re
from itertools import groupby
from operator import itemgetter
import pickle
import multiprocessing
import sys
import os
import pickle
import math

dns_or_doh = 'location.services.mozilla.com'
closeworldpath = '../hp131_cw/closed-world/'

domaindict = {}
with open('top-1m_tranco2021.csv', encoding = 'utf-8') as f:
    for l in f:
        l = l.split(',')
        domaindict[int(l[0])-1] = 'www.' + l[1].strip()

with open('excludedomains.data', 'rb') as f:
    excludedomain = pickle.load(f)

with open('indexmapping.data', 'rb') as f:
    indexmapping = pickle.load(f)

def separate_domains(dns, df, idx, num_pkts_in_one_file):
    # returns a list of (trace, label)
    first_pkts = dns['index'].to_numpy().tolist()
    front = 0
    separate = []
    if num_pkts_in_one_file == 12:
        num_pkts_in_one_file = 200
    else:
        num_pkts_in_one_file = 100
    pad_idx = (idx-1)*num_pkts_in_one_file
    for i, j in enumerate(first_pkts[1:]):
        rows = np.arange(front, j, 1)
        pkt = df.loc[rows]
        separate.append((pkt, pad_idx + i))
        front = j
    
    #insert last domain
    separate.append((df[front:len(df)], pad_idx + len(dns)-1))
    return separate

def get_src(df):
    return df[0][0]['ip.src'][0]

def get_dns(df):
    return df[df['_ws.col.Info'].str.endswith(dns_or_doh, na=False)]

count = 0
closeworld = os.listdir(closeworldpath)
closeworld.sort()


for directory in closeworld:
    dfs = []
    errors = []
    storage = []
    if os.path.isdir(closeworldpath+directory):
        print('DIR: ', directory, count, '/', len(os.listdir(os.getcwd())))
        count += 1
        l = len(os.listdir(closeworldpath+directory))
        for filename in os.listdir(closeworldpath+directory):
            if filename.endswith('.pickle'):
                try:
                    df = pd.read_pickle(closeworldpath+directory+'/'+filename)
                except Exception as e:
                    print(directory+'/'+filename)
                    print("ERROR : "+str(e))
                    errors.append(directory+'/'+filename + str(e))
                    continue
                df['index'] = np.arange(len(df))
                idx = int(filename.split('-')[1])
                dfs.append((df, l, int(directory), idx))
        for df, dirsize, directory, idx in dfs:
            dns = get_dns(df)
            separate = separate_domains(dns, df, idx, dirsize)
            src = get_src(separate)
            for inner_df, label in separate:
                print('label', label)
                if label >= 1000 or label in excludedomain:
                    continue
                zero = inner_df.index[0]
                for i in range(1, len(inner_df)):
                    inner_df.at[zero + i, '_ws.col.Time'] -= inner_df.at[zero, '_ws.col.Time']
                inner_df.at[zero, '_ws.col.Time'] = 0
                
                ipadd = inner_df[(inner_df['_ws.col.Protocol'].str.contains('DNS')) & (inner_df['_ws.col.Info'].str.contains('mozilla')) & (inner_df['_ws.col.Info'].str.contains('Standard query response'))]['_ws.col.Info'].tolist()
                c = set()
                for ip in ipadd:
                    ip = ip.split(' A ')
                    b = list(map(lambda x: re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",x), ip))
                    for i in b:
                        if i:
                            c.add(str(i.group(0)))
                
                inner_df = inner_df[(~inner_df['ip.src'].isin(c)) & (~inner_df['ip.dst'].isin(c)) & (~inner_df['_ws.col.Info'].str.contains('mozilla', na=False))]
                l = []
                l.append(indexmapping[label])
                lengths = inner_df['frame.len'].tolist()
                times = inner_df['_ws.col.Time'].tolist()
                direction = inner_df['ip.src'].tolist()
                direction = list(map(lambda x:-1 if x == src else 1, direction))
                tcp = list(map(lambda x:int(np.isnan(x)),inner_df['tcp.srcport'].tolist()))
                tcp = list(map(lambda x:-1 if x == 0 else 1, tcp))
                quic = list(map(lambda x:int(443 in x), inner_df[['udp.srcport', 'udp.dstport']].values.tolist()))
                quic = list(map(lambda x:-1 if x == 0 else 1, quic))
                data = inner_df['index'].tolist()
                burst = []
                for k, g in groupby(enumerate(data), lambda x: x[0]-x[1]):
                    le = len(list(map(itemgetter(1), g)))
                    burst.append(le)
                
                l.append(lengths)
                l.append(times)
                l.append(direction)
                l.append(tcp)
                l.append(quic)
                l.append(burst)
                storage.append(l)
        
    if not os.path.exists('./closedworld_data'):
        os.makedirs('./closedworld_data')
    with open('./closedworld_data/'+ str(directory)+'.pickle', "wb") as fh:
        pickle.dump(storage, fh)

