import numpy as np
import datetime
import json
import copy
from os import listdir
from os.path import isfile, join
now = lambda: datetime.datetime.now()

"""
Author : Avisena Abdillah Alwi

Created at : 9 Feb 2019
"""

class CharacterRecognition:

    cache_file_name = 'cache.json'

    def __init__(self, file_training='', alpha=1.0, threshold=1.0):
        """
        Konstruktor class, inisialisasi semua hal yang diperlukan
        :param file_training: path file training
        :param alpha: float
        :param threshold: ambang batas
        """
        self.__alpha = alpha
        self.__threshold = threshold
        self.__data_training_dict = self.__get_data_training(file_training)
        self.size = len(self.__data_training_dict)
        self.length = len(list(self.__data_training_dict.values())[0])

        self.__weight = {}
        for (key, value) in self.__data_training_dict.items():
            self.__weight[key] = [0] * self.length

        self.__bias = [0] * self.size

        self.__target = {}
        for index, (key, value) in enumerate(self.__data_training_dict.items()):
            self.__target[key] = [-1] * self.size
            self.__target[key][index] = 1

    def __get_data_training(self, file_training):
        """
        Mendapatkan data training dari file_training
        :param file_training:
        :return: dict dengan nilai bipolar. Contoh {"0":[1,1,-1,1....], "1":[1,1,-1,1....]}
        """
        file = open(file_training, 'r')
        data_dict = {}
        key_huruf = ''  # Key huruf yang saat ini diproses
        line_huruf = ''  # String builder yang isinya # dan . contoh "#.#...#.#.##....#.. ..."
        for one_line_in_file in file:
            # Hapus \n dari setiap baris yang dibaca oleh file
            line = one_line_in_file.rstrip()

            # Jika baris hanya 1 karakter, itu adalah KEY dari huruf yang diproses selanjutnya
            if len(line) == 1:
                key_huruf = line
                continue
            # Jika baris kosong, artinya proses ganti huruf
            if line == '':
                line_huruf_array = []
                for one_char in line_huruf:
                    line_huruf_array.append(one_char)
                data_dict[key_huruf] = line_huruf_array
                line_huruf = ''
                key_huruf = ''
            else:  # Jika tidak, gabungkan line yang sekarang dengan line_huruf
                line_huruf += line

        for (key, array_character) in data_dict.items():
            data_dict[key] = [self.__convert_to_bipolar(character) for character in array_character]
        file.close()
        return data_dict

    def __convert_to_bipolar(self, value='#', high_value='#'):
        """
        Mengkonversi ke bipolar
        :param value: nilai yang ingin diubah
        :param high_value: nilai yang nantinya menjadi 1
        :return: Int
        """
        if value == high_value:
            return 1
        else:
            return -1

    def __activation(self, x):
        """
        Fungsi aktivasi
        :param x: nilai
        :return: Int
        """
        if x > self.__threshold:
            return 1
        if x < -self.__threshold:
            return -1
        else:
            return 0

    def train(self, load_from_cache=False):
        """
        Method untuk melatih data.
        :param load_from_cache: load weight dan bias dari cache jika ada, jika tidak ada maka akan melakukan train
        :return: -
        """
        try:
            if load_from_cache:
                with open(self.cache_file_name, 'r') as file_cache:
                    json_object = json.load(file_cache)
                    print("Load weight from cache")
                    self.__weight = json_object['weight']
                    print("Load bias from cache")
                    self.__bias = json_object['bias']
                    return
        except FileNotFoundError:
            print("Cache file not found")
            print("Training continues")

        start_time = copy.deepcopy(now())
        print("Training started")
        stop = False
        epoch = 1
        while not stop:
            epoch += 1
            stop = True
            # Key disini adalah 1,2,3,4,5,6...A,B,...Z
            for (key, data) in self.__data_training_dict.items():
                # Perlu ditranspose karena butuh array dengan dimensi n size x n banyak huruf, sedangkan kita punya
                # n huruf x n size
                v = np.dot(data, np.transpose(list(self.__weight.values()))) + self.__bias
                y = [self.__activation(x) for x in v]
                # assign target dengan target angka/huruf yang diproses saat ini
                target = self.__target[key]

                for i in range(self.length):
                    # enumerate return 2 nilai, pertama index loop, kedua value pada list
                    for j, keyz in enumerate(list(self.__weight.keys())):
                        # update bias dan weight jika y berbeda dengan target
                        if y[j] != target[j]:
                            stop = False
                            self.__weight[keyz][i] = self.__weight[keyz][i] + self.__alpha * target[j] * data[i]
                            self.__bias[j] = self.__bias[j] + self.__alpha * target[j]
        # cetak keterangan tambahan
        print("Training finished in : " + str((now() - start_time).total_seconds()) + " second")
        print("Epoch : " + str(epoch))
        with open(self.cache_file_name, 'w') as file_cache:
            json.dump({"weight": self.__weight, 'bias': self.__bias}, file_cache)

    def test(self, file_name=""):
        """
        Menguji suatu data yang terdapat pada file_name
        :param file_name: path file data uji
        :return: karakter yang dideteksi
        """
        try:
            file = open(file_name, 'r')
        except FileNotFoundError:
            print("No File detected")
            return
        # Tahap ini mirip seperti pada method __get_data_training
        data_test = []
        line_char_buffer = ''
        for line in file:
            line = line.rstrip()
            if line != '':
                line_char_buffer += line
            if line == '':
                break
        for char in line_char_buffer:
            data_test.append(self.__convert_to_bipolar(char))
        file.close()

        list_of_recognized_char = list(self.__weight.keys())
        v = np.dot(data_test, np.transpose(list(self.__weight.values()))) + self.__bias
        y = np.array([])
        for x in v:
            y = np.append(y, self.__activation(x))
        try:
            # Kembalikan string karakter yang dikenali
            return list_of_recognized_char[np.where(y == 1)[0][0]]
        except IndexError:
            return "-"


# ==============================================================
# ==========================BEGIN===============================
# ==============================================================

cr = CharacterRecognition('dataset.txt', alpha=1, threshold=0.1)
cr.train(load_from_cache=True)

# f untuk setiap item pada folder "TestFile" jika f tersebut adalah sebuah file masukkan ke list
# pakai List Comprehensions python
fileUji = [f for f in listdir('TestFile') if isfile(join('TestFile', f))]
# Uji semua file uji pada folder TestFile
for fileName in fileUji:
    print("Mirip " + fileName[:1] + " (" + fileName + ") dikenali sebagai : ", end='')
    print(cr.test("TestFile/" + fileName))

# Untuk menguji 1 file panggil method test pada instance CharacterRecognition dan kirim path file uji
print("Mirip T dideteksi sebagai : ", end='')
print(cr.test("TestFile/T_test.txt"))

# ==============================================================
# ===========================END================================
# ==============================================================
