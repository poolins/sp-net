import random

size = 256  # размер S-блока
min_hamming_distance = 4  # минимальное расстояние Хэмминга


# вычисление расстояния Хэмминга между двумя числами
# d(x,y) - число символов, в которых числа x и y различаются
def hamming_distance(x, y):
    return bin(x ^ y).count('1')


def generate_s_box():
    number_of_errors = 0
    while True:  # пока не найдётся решение
        try:
            values = list(range(size))  # все возможные значения (0, 1, ..., size-1)
            s_box = [-1] * size

            # поиск значений с достаточным расстоянием Хэмминга
            for original in range(size):
                candidates = []
                for v in values:
                    if hamming_distance(original, v) >= min_hamming_distance:
                        candidates.append(v)

                if not candidates:
                    raise ValueError(
                        f"Не удалось найти подстановку для {original} с минимальным расстоянием {min_hamming_distance}")

                # случайный выбор из подходящих кандидатов
                substituted = random.choice(candidates)
                s_box[original] = substituted
                values.remove(substituted)  # удаление выбранного значения, чтобы обеспечить уникальность

            print(f'Количество повторных генераций для составления s-блока: {number_of_errors}')
            return s_box

        except ValueError:
            number_of_errors += 1
