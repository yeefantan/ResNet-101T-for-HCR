import sys
sys.path.append('../')
from utils.lib import *

def rename(file_path,file_ext):
    
    for item in file_ext:
        new_file_path = file_path+item
        dirs = os.listdir(new_file_path)
        if '.DS_Store' in dirs:
            dirs.remove('.DS_Store')
        for i,file in enumerate(dirs):
            new_file_name = new_file_path+'//'+str(i)+'.jpg'
            old_file_name = new_file_path+'//'+file
            os.rename(old_file_name,new_file_name)

def read_label(file='../src/labels/data.txt'):

    f = open(file,"r")
    full_label = list()
    for line in f:
        for i in range(7):
            f.readline()
        for line in f:
            full_label.append(line)


    real_label = list()
    for i in full_label:
        real_label.append(i[14:])

    final_label = list()
    for i in real_label:
        i = i.replace('\n','|')
        final_label.append(i)

    return final_label

def read_final_label(file='../src/labels/final_data.txt'):

    f = open(file,"r")
    full_label = list()
    for line in f:
        full_label.append(line)


    return full_label

def read_dict(file):

    f = open(file,"r")
    full_label = list()
    for line in f:
        full_label.append(line[:-2])


    return full_label

def encode_string(label,chars):

    text = unicodedata.normalize("NFKD", label).encode("ASCII", "ignore").decode("ASCII")
    text = " ".join(text.split())
    UNK_TK = "¤"
    groups = ["".join(group) for _, group in groupby(text)]
    encoded = []
    UNK = chars.find(UNK_TK)

    for item in text:
        index = chars.find(item)
        index = UNK if index == -1 else index
        encoded.append(index)
        
    return encoded