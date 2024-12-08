import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math_tests import bright_hist, grid_test, acf_test
from boxes_generating import generate_s_box

p_box = np.array(
    [11, 20, 8, 27, 30, 22, 12, 25, 23, 19,
     3, 13, 7, 6, 15, 24, 9, 18, 28, 5,
     16, 10, 4, 26, 31, 0, 2, 14, 29, 17,
     1, 21])

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
def apply_substitute(block, s_boxes, round):
    num_subblocks = NUM_OF_S_BLOCKS
    subblock_size = len(block) // num_subblocks
    ost = len(block) % num_subblocks

    # делим на подблоки
    subblocks = []
    for i in range(num_subblocks):
        subblock = block[i * subblock_size:(i + 1) * subblock_size]
        subblocks.append(subblock)

    if ost > 0:
        subblocks[-1] = np.concatenate((subblocks[-1], block[-ost:]))

    for i in range(NUM_OF_S_BLOCKS):
        if len(subblocks[i]) > 0:
            subblocks[i] = np.array([s_boxes[(i + (round * 4))][pixel] for pixel in subblocks[i]])

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
run_tests(brightness_array, "исходное изображение")

s_boxes = []
# шифрование
for round in range(ROUNDS):
    # разбиваем входной поток данных на блоки по 32 пикселя каждый
    data_blocks = []
    for i in range(0, len(brightness_vec), DATA_BLOCK):
        data_block = brightness_vec[i:i + DATA_BLOCK]
        data_blocks.append(data_block)

    # генерируем s блоки для данного раунда шифрования
    for _ in range(NUM_OF_S_BLOCKS):
        s_boxes.append(generate_s_box())

    print(f"Количество s блоков: {len(s_boxes)}")

    result_vec = []
    for block in data_blocks:
        new_block = apply_substitute(block, s_boxes, round)
        new_block = apply_permutation(new_block)
        result_vec.extend(new_block)

    brightness_vec = result_vec
    result_array = np.array(result_vec).reshape(-1, width)
    show_image(result_array, f"Изображение после {round + 1} раунда шифрования")
    run_tests(result_array, f"{i + 1} раунд шифрования")
