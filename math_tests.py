import numpy as np
import matplotlib.pyplot as plt


# тест на решетчатость
def grid_test(brightness_array, title):
    vec = brightness_array.flatten()  # преобразуем матрицу изображения в вектор

    # данные для графика
    data_x = vec[:-1]
    data_y = vec[1:]

    # построение графика
    plt.scatter(data_x, data_y, label="Точки", s=10)  # s — размер точек на графике
    plt.title(f"Тест на решетчатость: {title}")
    plt.xlabel("v[i]")
    plt.ylabel("v[i+1]")
    plt.legend()
    plt.grid(True)
    plt.show()


# гистограмма яркости пикселей
def bright_hist(brightness_array, gist_title):
    hist = [0] * 13  # 256 делим на 20 диапазонов

    # заполнение массива hist
    for y in range(len(brightness_array)):
        for x in range(len(brightness_array[0])):
            range_index = brightness_array[y][x] // 20
            hist[range_index] += 1

    # подготовка данных для построения графика
    range_labels = [f"{i * 20}..{min((i + 1) * 20 - 1, 255)}" for i in range(13)]

    # построение графика
    plt.figure(figsize=(10, 6))
    plt.bar(range_labels, hist, color='gray')
    plt.xlabel("Яркость")
    plt.ylabel("Количество пикселей")
    plt.title(f"Гистограмма яркости пикселей: {gist_title}")
    plt.xticks(rotation=45)
    plt.show()


def acf_test(brightness_array, title):
    vector = brightness_array.flatten()  # преобразование матрицы изображения в вектор
    centered_vector = center_vector(vector)  # центрирование вектора
    acf = calculate_acf(centered_vector)  # вычисление АКФ
    plot_acf(acf, title)  # построение графика


# центрирование вектора
def center_vector(vector):
    return vector - np.mean(vector)


# вычисление нормированной автокорреляционной функции с помощью БПФ
def calculate_acf(vector):
    N = len(vector)
    fft_vector = np.fft.fft(vector)  # прямое БПФ
    fft_conj_vector = np.conj(fft_vector)  # комплексное сопряжение
    acf_freq = fft_vector * fft_conj_vector  # произведение спектров
    acf = np.fft.ifft(acf_freq).real  # обратное БПФ для получения АКФ и оставляем только действительную часть
    acf = acf / acf[0]  # нормировка АКФ
    return acf[:N // 2]  # берем только половину для симметрии


# построение графика автокорреляционной функции
def plot_acf(acf, title):
    shifts = np.arange(len(acf))  # последовательные сдвиги
    plt.figure(figsize=(10, 5))
    plt.plot(shifts, acf, label="Автокорреляционная функция")
    plt.xlabel("Сдвиг (дискретное время)")
    plt.ylabel("Нормированное значение АКФ")
    plt.title(f"Автокорреляционная функция: {title}")
    plt.legend()
    plt.grid()
    plt.show()
