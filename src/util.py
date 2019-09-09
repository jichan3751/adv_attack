import os
import sys
import csv
import time
import smtplib
import json
import subprocess
from pprint import pprint

import numpy as np
##### Utils ######
class Timer:
    # use as:
    # with Timer() as t:

    def __init__(self, name=None, log = 1):
        self.job_name = name
        self.start = time.time()
        self.log = log

    def __enter__(self):
        # self.__init__(job_name)
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.log ==1:
            if self.job_name == None:
                print("Took %.2f seconds"%self.interval)
            else:
                print("%s Took %.2f seconds"%(self.job_name, self.interval))

class Logger:
    def __init__(self, fname):
        self.fname = fname
        self.file = open(fname,"w+")

    def log(self, str):
        print(str) # print to stdout
        print(str,file = self.file)

    def close(self):
        self.file.close()



def chunk(a, i, n):
    a2 = chunkify(a, n)
    return a2[i]

def chunkify(a, n):
    # splits list into even size list of lists
    # [1,2,3,4] -> [1,2], [3,4]

    k, m = divmod(len(a), n)
    gen = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(gen)

# def chunkify(a, n):
#     # splits list into even size list of lists
#     # [1,2,3,4] -> [1,3], [2,4]
#     # [1,2,3,4,5] -> [1,3,5], [2,4]

#     ch = [[] for _ in range(n)]

#     for i in range(len(a)):
#         a_i = a[i]
#         # k, r = divmod(i, n)
#         r = i % n
#         ch[r].append(a_i)

#     return ch


def array_inv(p):
    # p = [1,2,4,0,3]
    p_inv = [-1 for _ in range(len(p))]

    for i in range(len(p)):
        p_inv[p[i]] = i

    return p_inv


### for iterating experiments ######

# better name?
def iteration_tuples(arrays):
    return list(itertools.product(*arrays))

def iter_core(arrays, rank, n_node):
    ll1 = iteration_tuples(arrays)
    ll2 = chunkify(ll1, n_node)
    return ll2[rank]

def dict_product(dicts): # python3
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def iter_core_dict(dict_arrays, rank, n_node):
    ll1 = dict_product(dict_arrays)
    ll2 = chunkify(ll1, n_node)
    return ll2[rank]


def generate_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_dirs(directorys):
    for directory in directorys:
        generate_dir(directory)

def read_array_col(fname, dtype = None):
    # read array column direction
    list0 = []
    csv_file = open(fname, "r")
    cr = csv.reader(csv_file , delimiter=',', quotechar='|')
    for row in cr:

        if dtype == "int":
            row1 = int(row[0])
        elif dtype == "float":
            row1 = float(row[0])
        else:
            row1 = row[0] # as string

        list0.append(row1)
    csv_file.close()
    return list0

def save_array_col(fname, list0):
    # save array column direction
    csv_file = open(fname, "w")
    cw = csv.writer(csv_file , delimiter=',', quotechar='|')
    for i in range(len(list0)):
        cw.writerow([str(list0[i])])
    csv_file.close()


# returns list of n_test True s and num_data-n_test Falses
def nCr_lookup(num_data,n_test):
    test_lookup = [False for _ in range(num_data)]
    if num_data < n_test:
        test_ind = np.array([])
    else:
        test_ind = np.random.choice(num_data, n_test,replace = False)

    for j in test_ind:
        test_lookup[j] = True

    return test_lookup


def sort_array_with_max_value(seq0,max0):
    seq1 = [0 for _ in range(max0)]
    for i0 in range(len(seq0)):
        seq1[seq0[i0]] = 1
    seq2 = [i1 for i1 in range(len(seq1)) if seq1[i1]==1]  # sorted list
    seq1 = None
    return seq2

def save_list_of_tuples(fname, L):
    csv_file = open(fname, "w")
    cw = csv.writer(csv_file , delimiter=',', quotechar='|')
    for i in range(len(L)):
        cw.writerow([str(j) for j in list(L[i])])
    csv_file.close()

def save_array(fname, L):
    csv_file = open(fname, "w")
    cw = csv.writer(csv_file , delimiter=',', quotechar='|')
    for i in range(L.shape[0]):
        cw.writerow([str(j) for j in list(L[i])])
    csv_file.close()


def send_mail(msg):
    # send mail for notification usage
    GMAIL_ADDRESS = "jichan3751notification@gmail.com"
    GMAIL_PW = "notification" # i just use this for notification so don't worry
    SEND_FROM = GMAIL_ADDRESS
    SEND_TO = GMAIL_ADDRESS

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(GMAIL_ADDRESS, GMAIL_PW)

    # print("login success")
    server.sendmail(SEND_FROM, SEND_TO, msg)
    server.quit()
    print("msg %s send by mail complete"%msg)

def write_json( fname, dict0,sort = False):
    str0 = json.dumps(dict0, sort_keys=sort, indent=4, separators=(',', ': '))
    with open(fname,'w') as file:
        file.write(str0)

def read_json(fname):
    with open(fname) as f:
        data = json.load(f)
    return data


def merge_eps_to_pdf(filenames,outname):

    # files = []
    # for test_rate in loop['test_rates']:
    #     for rank in loop['ranks']:
    #         for dcy in loop['dcys']:
    #             for initial_step_size in loop['initial_step_sizes']:
    #                 fname = "./plot/jf_synthetic_tr%.2f_n20_mu0.005_r%d_st%.2f_dcy_%.2f_errors.eps" % \
    #                     (test_rate,rank,-initial_step_size,dcy)
    #                 if os.path.isfile(fname):
    #                     files.append(fname)
    #                 else:
    #                     print('skipping %s since file does not exist'%(fname))


    cmd = ["gs","-sDEVICE=pdfwrite", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dEPSCrop", "-sOutputFile=%s"%(outname)]
    cmd = cmd + filenames
    subprocess.call(cmd)



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("---Program Ended in %0.2f hour " % ((time.time() - start_time)/float(3600)))
