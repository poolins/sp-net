import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math_tests import bright_hist, grid_test, acf_test

s_box_1 = [200, 26, 84, 87, 120, 203, 206, 185, 196, 248, 233, 122, 117, 160, 70, 238, 0, 29, 205, 23, 125, 254, 83, 7,
           133, 61, 63, 255, 216, 47, 36, 184, 188, 177, 22, 73, 232, 112, 190, 60, 118, 67, 215, 143, 16, 212, 45, 14,
           152, 109, 25, 49, 231, 219, 181, 247, 220, 201, 42, 115, 114, 108, 43, 191, 126, 9, 2, 88, 71, 213, 244, 139,
           142, 223, 68, 236, 222, 19, 40, 193, 224, 72, 44, 41, 116, 173, 11, 82, 58, 208, 151, 245, 64, 165, 92, 4,
           186, 113, 110, 246, 169, 174, 48, 253, 155, 97, 251, 105, 207, 51, 249, 199, 202, 228, 127, 179, 85, 159, 5,
           195, 62, 96, 17, 79, 106, 218, 252, 243, 38, 69, 37, 182, 217, 8, 53, 187, 86, 24, 147, 89, 227, 55, 239,
           150, 57, 75, 198, 30, 130, 148, 229, 123, 107, 134, 197, 91, 241, 156, 104, 141, 250, 28, 172, 242, 167, 180,
           6, 237, 59, 135, 140, 10, 65, 214, 149, 192, 121, 50, 176, 32, 234, 163, 138, 78, 145, 103, 76, 209, 210,
           226, 1, 170, 161, 98, 235, 18, 146, 13, 101, 35, 34, 95, 94, 54, 194, 211, 144, 100, 128, 27, 175, 124, 204,
           15, 81, 102, 153, 90, 21, 189, 154, 230, 33, 20, 136, 77, 3, 157, 158, 52, 183, 12, 93, 168, 111, 221, 164,
           171, 74, 225, 66, 137, 99, 178, 80, 131, 31, 132, 129, 162, 119, 56, 39, 166, 240, 46]
s_box_2 = [173, 90, 169, 103, 239, 126, 192, 69, 156, 148, 45, 55, 153, 70, 139, 74, 61, 249, 138, 122, 25, 232, 98,
           134, 73, 188, 15, 246, 167, 254, 27, 41, 112, 44, 143, 20, 140, 0, 238, 1, 40, 80, 79, 136, 255, 233, 26,
           197, 77, 87, 247, 120, 28, 47, 244, 121, 200, 49, 9, 118, 142, 31, 81, 65, 116, 248, 38, 147, 224, 108, 83,
           205, 57, 22, 209, 234, 99, 190, 207, 13, 30, 145, 203, 95, 221, 175, 106, 161, 105, 104, 185, 171, 170, 228,
           62, 66, 54, 82, 132, 114, 5, 68, 43, 124, 150, 187, 193, 107, 163, 225, 213, 215, 100, 189, 16, 117, 92, 177,
           162, 3, 211, 135, 152, 151, 93, 160, 191, 123, 46, 253, 97, 127, 119, 63, 24, 214, 252, 166, 216, 19, 240,
           12, 59, 36, 154, 102, 186, 130, 172, 75, 144, 39, 33, 237, 67, 14, 58, 85, 212, 129, 146, 125, 76, 195, 199,
           184, 86, 226, 110, 131, 236, 141, 180, 219, 137, 174, 101, 109, 202, 220, 210, 56, 17, 227, 88, 222, 196,
           235, 251, 29, 64, 242, 35, 111, 84, 18, 168, 11, 204, 53, 206, 8, 133, 217, 223, 7, 4, 10, 34, 60, 165, 128,
           229, 91, 37, 241, 159, 231, 52, 72, 176, 50, 201, 158, 21, 113, 178, 182, 157, 230, 155, 78, 179, 245, 23,
           250, 42, 164, 218, 96, 194, 89, 48, 115, 32, 181, 243, 149, 2, 183, 51, 6, 71, 94, 198, 208]
s_box_3 = [110, 55, 226, 96, 44, 84, 238, 27, 48, 124, 51, 230, 247, 42, 38, 189, 131, 140, 8, 13, 143, 186, 173, 69, 9,
           155, 127, 206, 184, 204, 36, 45, 106, 108, 164, 218, 198, 65, 82, 209, 50, 54, 98, 93, 220, 122, 166, 99,
           255, 1, 160, 64, 117, 74, 25, 18, 137, 193, 229, 31, 228, 95, 70, 109, 169, 56, 123, 182, 19, 158, 22, 128,
           14, 46, 5, 197, 139, 77, 181, 23, 88, 87, 121, 134, 112, 221, 85, 151, 6, 199, 3, 115, 153, 12, 90, 52, 202,
           145, 66, 60, 148, 16, 233, 33, 149, 76, 249, 212, 241, 75, 195, 232, 167, 190, 11, 92, 35, 68, 157, 4, 162,
           156, 43, 53, 10, 24, 144, 57, 203, 222, 210, 104, 170, 94, 20, 175, 81, 105, 254, 67, 63, 248, 118, 138, 239,
           240, 100, 142, 250, 83, 30, 107, 214, 89, 211, 178, 205, 2, 200, 91, 79, 119, 150, 246, 165, 26, 224, 227,
           126, 116, 73, 41, 141, 103, 207, 180, 219, 86, 168, 251, 252, 194, 201, 196, 176, 146, 208, 171, 235, 114,
           125, 245, 15, 163, 237, 159, 187, 111, 152, 37, 244, 102, 17, 216, 71, 47, 234, 192, 61, 28, 72, 136, 62, 80,
           113, 179, 39, 135, 174, 101, 0, 29, 7, 120, 242, 223, 154, 34, 177, 225, 58, 129, 253, 21, 188, 32, 191, 49,
           147, 213, 217, 40, 243, 132, 215, 97, 130, 78, 133, 231, 172, 161, 185, 183, 59, 236]
s_box_4 = [21, 4, 132, 94, 234, 197, 198, 68, 27, 127, 148, 89, 13, 150, 78, 177, 129, 135, 196, 157, 116, 226, 251, 86,
           164, 176, 248, 83, 44, 170, 95, 15, 113, 107, 5, 169, 42, 62, 122, 211, 142, 222, 195, 233, 105, 152, 231,
           43, 219, 33, 123, 133, 22, 205, 192, 243, 57, 93, 125, 240, 220, 51, 250, 184, 92, 88, 194, 0, 2, 121, 36,
           63, 163, 217, 199, 66, 228, 58, 60, 193, 76, 75, 239, 147, 155, 183, 245, 230, 47, 128, 120, 61, 145, 109,
           223, 11, 45, 70, 154, 161, 80, 187, 175, 134, 162, 64, 7, 119, 236, 188, 117, 246, 118, 52, 208, 108, 229,
           212, 146, 210, 19, 214, 17, 59, 159, 241, 247, 249, 84, 25, 97, 180, 55, 167, 54, 37, 244, 202, 41, 238, 204,
           46, 71, 200, 12, 1, 96, 103, 191, 216, 153, 207, 144, 253, 91, 38, 115, 77, 112, 182, 69, 111, 24, 48, 178,
           156, 190, 34, 131, 235, 124, 90, 185, 225, 3, 168, 23, 49, 85, 171, 16, 227, 149, 10, 104, 106, 139, 181, 72,
           130, 74, 40, 138, 53, 136, 137, 255, 254, 224, 172, 141, 151, 20, 140, 237, 14, 87, 26, 179, 209, 32, 206,
           98, 30, 158, 31, 174, 221, 35, 39, 110, 143, 29, 173, 79, 9, 114, 215, 6, 232, 81, 186, 56, 100, 213, 252,
           73, 101, 242, 82, 67, 201, 218, 50, 18, 126, 165, 99, 189, 8, 166, 160, 65, 203, 28, 102]

p_box = np.array(
    [11, 20, 8, 27, 30, 22, 12, 25, 23, 19,
     3, 13, 7, 6, 15, 24, 9, 18, 28, 5,
     16, 10, 4, 26, 31, 0, 2, 14, 29, 17,
     1, 21])

# обратные таблицы подстановки для расшифровки
inv_s_box_1 = [0] * 256
inv_s_box_2 = [0] * 256
inv_s_box_3 = [0] * 256
inv_s_box_4 = [0] * 256

for i in range(256):
    inv_s_box_1[s_box_1[i]] = i
    inv_s_box_2[s_box_2[i]] = i
    inv_s_box_3[s_box_3[i]] = i
    inv_s_box_4[s_box_4[i]] = i

# выбор изображения
image_path = 'images/moon64x64.jpg'
# image_path = 'images/small.jpg'
# image_path = 'images/field500x500.jpg'
# image_path = 'images/p236x236.jpg'

im = Image.open(image_path)
(width, height) = im.size
print(f"Размер изображения: {width}x{height}")

# размер s-блока
S_BLOCK_SIZE = 8
NUM_OF_S_BLOCKS = 4

# размер блока входных данных
DATA_BLOCK = 32

# количество раундов шифрования
ROUNDS = 1


# получение изображения из массива яркости пикселей
def show_image(brightness_array, title):
    image = Image.fromarray(np.uint8(brightness_array), 'L')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.gcf().canvas.manager.set_window_title(title)
    plt.show()


# преобразование изображения в матрицу яркости пикселей
def load_image_as_array(path):
    image = Image.open(path).convert("L")  # конвертация в черно-белое
    image_array = np.array(image)  # <- двумерный массив HxW
    return image_array


# функция применения S-блока
# вход: блок данных 32 пикселя
def apply_substitute(block):
    num_subblocks = 4
    subblock_size = len(block) // num_subblocks
    ost = len(block) % num_subblocks

    # делим на подблоки
    subblocks = []
    for i in range(num_subblocks):
        subblock = block[i * subblock_size:(i + 1) * subblock_size]
        subblocks.append(subblock)

    if ost > 0:
        subblocks[-1] = np.concatenate((subblocks[-1], block[-ost:]))

    if len(subblocks[0]) > 0:
        subblocks[0] = np.array([s_box_1[pixel] for pixel in subblocks[0]])

    if len(subblocks[1]) > 0:
        subblocks[1] = np.array([s_box_2[pixel] for pixel in subblocks[1]])

    if len(subblocks[2]) > 0:
        subblocks[2] = np.array([s_box_3[pixel] for pixel in subblocks[2]])

    if len(subblocks[3]) > 0:
        subblocks[3] = np.array([s_box_4[pixel] for pixel in subblocks[3]])

    merge_subblocks = np.concatenate(subblocks)
    return merge_subblocks


# функция перестановки
def apply_permutation(block):
    effective_p_box = []
    for i in p_box:
        if i < len(block):
            effective_p_box.append(i)
    effective_p_box = np.array(effective_p_box)
    permuted_block = block[effective_p_box]
    return permuted_block


# ************** для расшифровывания **************
# функция обратной перестановки
def apply_inv_permutation(block):
    block = np.array(block)
    # массив для обратной перестановки
    inv_p_box = np.zeros(len(block), dtype=int)
    effective_p_box = []
    for i in p_box:
        if i < len(block):
            effective_p_box.append(i)
    effective_p_box = np.array(effective_p_box)

    for i, val in enumerate(effective_p_box):
        inv_p_box[val] = i

    inverse_permuted_block = block[inv_p_box]
    return inverse_permuted_block


# функция обратного применения s блока
def apply_inv_substitute(block):
    num_subblocks = 4
    subblock_size = len(block) // num_subblocks
    ost = len(block) % num_subblocks

    # делим на подблоки
    subblocks = []
    for i in range(num_subblocks):
        subblock = block[i * subblock_size:(i + 1) * subblock_size]
        subblocks.append(subblock)

    if ost > 0:
        subblocks[-1] = np.concatenate((subblocks[-1], block[-ost:]))

    if len(subblocks[0]) > 0:
        subblocks[0] = np.array([inv_s_box_1[pixel] for pixel in subblocks[0]])

    if len(subblocks[1]) > 0:
        subblocks[1] = np.array([inv_s_box_2[pixel] for pixel in subblocks[1]])

    if len(subblocks[2]) > 0:
        subblocks[2] = np.array([inv_s_box_3[pixel] for pixel in subblocks[2]])

    if len(subblocks[3]) > 0:
        subblocks[3] = np.array([inv_s_box_4[pixel] for pixel in subblocks[3]])

    merge_subblocks = np.concatenate(subblocks)
    return merge_subblocks


def run_tests(array, round_name):
    bright_hist(array, round_name)
    grid_test(array, round_name)
    acf_test(array, round_name)


# получаем массив значения яркости пикселей
brightness_array = load_image_as_array(image_path)  # <-- двумерный массив
brightness_vec = brightness_array.flatten()  # <-- одномерный массив (вектор)

# разбиваем входной поток данных на блоки по 32 пикселя каждый
data_blocks = []
for i in range(0, len(brightness_vec), DATA_BLOCK):
    data_block = brightness_vec[i:i + DATA_BLOCK]
    data_blocks.append(data_block)

show_image(brightness_array, "Исходное изображение")
#run_tests(brightness_array, "исходное изображение")

# шифрование
for i in range(ROUNDS):
    result_vec = []
    for block in data_blocks:
        new_block = apply_substitute(block)
        new_block = apply_permutation(new_block)
        result_vec.extend(new_block)

    result_array = np.array(result_vec).reshape(-1, width)
    show_image(result_array, f"Изображение после {i + 1} раунда шифрования")
    #run_tests(result_array, f"{i + 1} раунд шифрования")

# расшифровывание
data_blocks_enc = []
for i in range(0, len(result_vec), DATA_BLOCK):
    data_block = result_vec[i:i + DATA_BLOCK]
    data_blocks_enc.append(data_block)

for i in range(ROUNDS):
    decrypted_vec = []
    # последовательное обратное преобразование каждого блока
    for block in data_blocks_enc:
        new_block = apply_inv_permutation(block)
        new_block = apply_inv_substitute(new_block)
        decrypted_vec.extend(new_block)

    # преобразуем обратно в матрицу изображения
    decrypted_array = np.array(decrypted_vec).reshape(-1, width)
    show_image(decrypted_array, "Изображение после расшифрования")
