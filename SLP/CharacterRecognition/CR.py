import operator

import numpy as np


class CR:
    def __init__(self, file_training='', alpha=1, treshold=1):
        self.__alpha = alpha
        self.__treshold = treshold
        self.__data_training_dict = self.__get_data_training(file_training)
        self.size = len(self.__data_training_dict)
        self.length = len(list(self.__data_training_dict.values())[0])

        self.__weight = {}
        for (key, value) in self.__data_training_dict.items():
            self.__weight[key] = [0.5] * self.length

        self.__bias = [0.5] * self.size

        self.target = {}
        index = 0
        for (key, value) in self.__data_training_dict.items():
            self.target[key] = [-1] * self.size
            self.target[key][index] = 1
            index += 1

    def __get_data_training(self, file_training):
        file = open(file_training, 'r')
        data_dict = {}
        key_huruf = ''
        line_huruf = ''
        for one_line_in_file in file:
            # Remove \n from
            line = one_line_in_file.rstrip()
            if len(line) == 1:
                key_huruf = line
                continue
            if line == '':
                line_huruf_array = []
                for one_char in line_huruf:
                    line_huruf_array.append(one_char)
                data_dict[key_huruf] = line_huruf_array
                line_huruf = ''
                key_huruf = ''
            else:
                line_huruf += line

        for (key, array_character) in data_dict.items():
            data_dict[key] = [self.__convert_to_bipolar(character) for character in array_character]
        file.close()
        return data_dict

    def __convert_to_bipolar(self, value='.', low_value='.'):
        if value == low_value:
            return -1
        else:
            return 1

    def __aktivasi(self, x):
        if x > self.__treshold:
            return 1
        if x < -self.__treshold:
            return -1
        else:
            return 0

    def train(self):
        # update __weight value
        stop = False
        index = 0
        while not stop and index <= 10:
            stop = True
            for (key, data) in self.__data_training_dict.items():
                v = np.dot(data, self.__weight[key]) + self.__bias
                y = []
                for x in v:
                    y.append(self.__aktivasi(x))

                for i in range(self.size):
                    for j in range(self.length):
                        oo = y[i]
                        ooo = self.target[key][i]
                        if oo != ooo:
                            stop = False
                            self.__weight[key][j] = self.__weight[key][j] + self.__alpha * self.target[key][i] * data[j]
                            self.__bias[i] = self.__bias[i] + self.__alpha * self.target[key][i]
            index += 1

    def test(self, file_name=""):
        if not file_name:
            print("No test file")
            return
        file = open(file_name, 'r')
        line_char = ''
        for line in file:
            line = line.rstrip()
            if line != '':
                line_char += line
            if line == '':
                break
        data_test = []
        for char in line_char:
            data_test.append(self.__convert_to_bipolar(char))

        file.close()
        hasil = {}
        for key, __weight1Huruf in self.__weight.items():
            v = np.sum(np.dot(data_test, __weight1Huruf) + self.__bias)
            hasil[key] = v
        # print(hasil)
        print(max(hasil.items(), key=operator.itemgetter(1))[0])


cr = CR('dataset.txt', alpha=1, treshold=1)
cr.train()

print("A dideteksi sebagai : ", end='')
cr.test('A_test.txt')

print("B dideteksi sebagai : ", end='')
cr.test('B_test.txt')

print("C dideteksi sebagai : ", end='')
cr.test('D_test.txt')

print("D dideteksi sebagai : ", end='')
cr.test('D_test.txt')

print("E dideteksi sebagai : ", end='')
cr.test('E_test.txt')

print("F dideteksi sebagai : ", end='')
cr.test('F_test.txt')

print("G dideteksi sebagai : ", end='')
cr.test('G_test.txt')
