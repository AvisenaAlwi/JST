import pandas as pd
import numpy as np

data = np.array(pd.read_csv('data.csv'))
data_test = np.array(pd.read_csv('data_uji.csv'))
data_train = np.array(pd.read_csv('data_latih.csv'))

print("Panjang data = %d " % len(data))
print("Panjang data uji + data latih = %d" % len(np.concatenate((data_test, data_train))))

test_is_valid = True
for idx, one_data_test in enumerate(data_test):
    valid = True
    for one_data_train in data_train:
        valid = not np.array_equal(one_data_test, one_data_train)
        if not valid:
            break

    if valid:
        print("Data uji ke-%d Valid" % (idx+1))
    else:
        print("Data uji ke-%d Tidak Valid" % (idx+1))
        test_is_valid = False
if test_is_valid:
    print("Semua data uji valid")
else:
    print("Data uji tidak valid")
