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

domaindict = {}
with open('top-1m_tranco2021.csv', encoding = 'utf-8') as f:
    for l in f:
        l = l.split(',')
        domaindict[int(l[0])-1] = 'www.' + l[1].strip()

excludedomain = [3, 4, 7, 9, 11, 12, 13, 15, 16, 17, 19, 21, 23, 25, 27, 28, 29, 31, 32, 36, 41, 42, 53, 56, 57, 58, 59, 62, 64, 66, 67, 69, 70, 72, 77, 82, 84, 86, 88, 90, 92, 96, 97, 101, 105, 106, 111, 112, 113, 114, 115, 117, 118, 121, 124, 126, 130, 131, 136, 139, 141, 142, 144, 147, 155, 160, 165, 170, 171, 172, 187, 189, 190, 199, 203, 205, 212, 217, 219, 220, 223, 224, 227, 232, 243, 245, 246, 247, 250, 252, 261, 266, 269, 274, 279, 280, 284, 285, 292, 294, 305, 309, 313, 314, 315, 319, 320, 327, 330, 332, 339, 348, 351, 352, 354, 360, 362, 363, 367, 369, 371, 373, 376, 379, 385, 390, 391, 395, 397, 400, 403, 404, 412, 420, 425, 428, 432, 441, 443, 448, 455, 456, 464, 465, 467, 469, 477, 478, 480, 481, 485, 492, 502, 504, 506, 507, 509, 516, 524, 526, 532, 533, 537, 541, 543, 545, 546, 558, 561, 565, 578, 583, 584, 589, 594, 596, 607, 611, 612, 626, 630, 636, 638, 639, 645, 647, 648, 653, 654, 661, 664, 670, 673, 682, 693, 699, 701, 704, 705, 708, 713, 719, 724, 730, 734, 737, 738, 741, 742, 747, 750, 753, 758, 759, 761, 767, 770, 783, 793, 797, 801, 802, 804, 809, 814, 822, 824, 826, 828, 829, 830, 848, 855, 856, 857, 858, 860, 863, 873, 874, 877, 882, 883, 886, 890, 895, 899, 902, 914, 920, 921, 923, 928, 940, 946, 967, 969, 970, 971, 978, 980, 982, 984]

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
closeworld = os.listdir('../hp131_cw/closed-world/')
closeworld.sort()


for directory in closeworld:
    dfs = []
    errors = []
    if os.path.isdir(directory):
        print('DIR: ', directory, count, '/', len(os.listdir(os.getcwd())))
        count += 1
        l = len(os.listdir(directory))
        for filename in os.listdir(directory):
            if filename.endswith('.pickle'):
                try:
                    df = pd.read_pickle(directory+'/'+filename)
                except Exception as e:
                    print(directory+'/'+filename)
                    print("ERROR : "+str(e))
                    errors.append(directory+'/'+filename + str(e))
                    continue
                df['index'] = np.arange(len(df))
                idx = int(filename[8])
                if filename[8:10]=='10':
                    idx = 10
                dfs.append((df, l, int(directory), idx))
        storage = []
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
                l.append(label)
                lengths = inner_df['frame.len'].tolist()
                times = inner_df['_ws.col.Time'].tolist()
                direction = inner_df['ip.src'].tolist()
                direction = list(map(lambda x:-1 if x == src else 1, direction))
                tcp = list(map(lambda x:int(np.isnan(x)),inner_df['tcp.srcport'].tolist()))
                tcp = list(map(lambda x:-1 if x == 0 else 1), tcp)
                quic = list(map(lambda x:int(443 in x), inner_df[['udp.srcport', 'udp.dstport']].values.tolist()))
                quic = list(map(lambda x:-1 if x == 0 else 1), quic)
                data = inner_df['index'].tolist()
                burst = []
                for k, g in groupby(enumerate(data), lambda x: x[0]-x[1]):
                    le = len(list(map(itemgetter(1), g)))
                    burst.append(le))
                        incom = True
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

