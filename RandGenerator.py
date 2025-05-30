import random

# 0'dan 24865'e kadar olan sayıların listesi
numbers = list(range(24866))

# Sayıları karıştır
random.shuffle(numbers)

# Dosya adı
file_name = "randInd24866.txt"

# Dosyayı yazma modunda aç
with open(file_name, "w") as file:
    # Her sayıyı dosyaya yaz
    for number in numbers:
        file.write(str(number) + "\n")


