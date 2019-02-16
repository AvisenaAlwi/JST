<p align="center">

![Python](https://www.andreabacciu.com/wp-content/uploads/2015/02/Python-Logo-PNG-Image.png)

</p>

## Single Layer Perceptron
### Optical Character Recognition
Program ini cukup mampu untuk mengenali angka 0-9 dan A-Z. Dengan data latih dan data uji
yang telah ditetapkan. Selain itu program juga dapat membuat/menyimpan weight dan bias kedalam
cache sehingga ketika program dijalankan kembali tidak perlu melakukan training lagi
dengan cara menambahkan argument ```load_from_cache = True``` pada method ```train()```, 
atau Anda memaksa program untuk train ulang dengan memanggil method ```train()``` tanpa argumen
atau Anda kirim nilai False ```train(False)```.

##### Hal yang perlu diperhatikan
1. Pada file ```dataset.txt``` setiap huruf diawali dengan karakter huruf tersebut 
lalu diikuti dengan bentuk huruf tersebut
2. Setiap huruf/angka dipisahkan dengan 1 baris kosong 
3. Pada bagian bawah file ```dataset.txt``` harus ada setidaknya 2 baris kosong
4. Pada file ```CharacterRecognition.py``` nilai alpha dan treshold bisa di ubah ubah sesuai kebutuhan,
akan tetapi mempengaruhi hasil akhir
5. Untuk file uji terdapat pada folder ```TestTile```, silahkan tambah atau modifikasi
file file dalam folder tersebut tersebut. **Ingat setiap file harus hanya 1 data uji saja.
Dan akhir dari file setikdanya ada 2 baris kosong.**
6. Anda bisa menambahkan dataset latih baru dengan menambahkannya pada file ```dataset.txt```
dengan memperhatikan poin ke 1, 2, dan 3, atau bisa lihat contoh di file ```dataset.txt```

##### Requirement
1. Python 3.x
2. Numpy

## License
```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```  