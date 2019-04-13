import pandas as pd
import numpy as np
import random

file = 'data.csv'

file_output_latih = 'data_latih.csv'
file_output_uji = 'data_uji.csv'

data_latih_in_persen = 70

raw_data = pd.read_csv(file)
columns = raw_data.columns.values
header = ""
for column in columns:
    header+=column+","
header = header[:-1]
header += "\n"

buffer_string_latih = header
buffer_string_uji = header

for kategori in range(1,14):
    one_class = np.array(raw_data[(raw_data.class_ == kategori)])
    jumlah_data_latih = int( len(one_class) * data_latih_in_persen / 100 )
    index_yang_dipakai_data_latih = []
    index = 1
    while index <= jumlah_data_latih:
        idx = random.randrange(0, jumlah_data_latih)
        while idx in index_yang_dipakai_data_latih:
            idx = random.randrange(0, jumlah_data_latih)

        line = ''
        for fitur in one_class[idx]:
            line+="{},".format(fitur)
        buffer_string_latih += line[:-1]
        buffer_string_latih += "\n"
        index_yang_dipakai_data_latih.append(idx)
        index+=1

    for index in range(len(one_class)):
        if index not in index_yang_dipakai_data_latih:
            line = ''
            for fitur in one_class[index]:
                line += "{},".format(fitur)
            buffer_string_uji += line[:-1]
            buffer_string_uji += "\n"

with open(file_output_latih, 'w') as file:
    buffer_string_latih = buffer_string_latih[:-1]
    file.write(buffer_string_latih)
with open(file_output_uji, 'w') as file:
    buffer_string_uji = buffer_string_uji[:-1]
    file.write(buffer_string_uji)
