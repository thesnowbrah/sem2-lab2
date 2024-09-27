from PIL import Image
import numpy as np

#Приведение изображения к raw формату

def imgtoraw(img):
    imgar = np.array(img)
    return imgar

#Конвертация RGB в YCbCr

def rgbtoycbcr(ar):
    y = np.zeros(len(ar) * len(ar[0]))
    Cb = np.zeros(len(ar) * len(ar[0]))
    Cr = np.zeros(len(ar) * len(ar[0]))
    for i in range(len(ar)):
        for j in range(len(ar[0])):
            y[j + i * len(ar[0])] = ar[i][j][0] * 0.299 + 0.587 * ar[i][j][1] + 0.114 * ar[i][j][2] + 16
            Cb[j + i * len(ar[0])] = ar[i][j][0] * (-0.148) - 0.29 * ar[i][j][1] + 0.439 * ar[i][j][2] + 128
            Cr[j + i * len(ar[0])] = ar[i][j][0] * 0.439 - 0.367 * ar[i][j][1] + 0.07 * ar[i][j][2] + 128
    return y, Cb, Cr




#Конвертация YCbCr в RGB


def ycbcrtorgb(ar):
    r = np.zeros(len(ar) * len(ar[0]) * 3)
    r = np.reshape(r, (len(ar), len(ar[0]), 3))
    for i in range(len(ar)):
        for j in range(len(ar[0])):
            r[i][j][0] = (ar[i][j][0] - 16) * 70406500 / 91008499 - (ar[i][j][1] - 128) * 41464000 / 91008499 + (
                        ar[i][j][2] - 128) * 145376500 / 91008499
            if r[i][j][0] < 0: r[i][j][0] = 0
            if r[i][j][0] > 255: r[i][j][0] = 255
            r[i][j][1] = (ar[i][j][0] - 16) * 101540500 / 91008499 - (ar[i][j][1] - 128) * 14558000 / 91008499 - (
                        ar[i][j][2] - 128) * 74066500 / 91008499
            if r[i][j][1] < 0: r[i][j][1] = 0
            if r[i][j][1] > 255: r[i][j][1] = 255
            r[i][j][2] = (ar[i][j][0] - 16) * 90813000 / 91008499 + (ar[i][j][1] - 128) * 183713000 / 91008499 + (
                        ar[i][j][2] - 128) * 83000 / 91008499
            if r[i][j][2] < 0: r[i][j][2] = 0
            if r[i][j][2] > 255: r[i][j][2] = 255
            # r[i][j][0] = (ar[i][j][0]-16) * 0.774 - 0.456 * (ar[i][j][1]-128) + 1.597 * (ar[i][j][2]-128)+10
            # r[i][j][1] = ar[i][j][0] - 0.344 * (ar[i][j][1] - 128) - 0.714 * (ar[i][j][2] - 128)
            # r[i][j][2] = ar[i][j][0] + 1.772 * (ar[i][j][1] - 128) + 0.00000041 * (ar[i][j][2] - 128)
    return r





#Обход матрицы зигзагом

def zigzag_order(matrix):
    rows, cols = matrix.shape
    result = np.zeros(rows * cols, dtype=int)
    index = -1
    bound = rows + cols - 1
    for i in range(bound):
        if i % 2 == 0:
            r = i if i < rows else rows - 1
            c = 0 if i < rows else i - rows + 1
            while r >= 0 and c < cols:
                index += 1
                result[index] = matrix[r, c]
                r -= 1
                c += 1
        else:
            r = 0 if i < cols else i - cols + 1
            c = i if i < cols else cols - 1
            while r < rows and c >= 0:
                index += 1
                result[index] = matrix[r, c]
                r += 1
                c -= 1
    return result


# matrix = [
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12]
# ]
#
# print(zigzag_order(np.array(matrix)))


#Обратная сборка матрицы после обхода зигзагом


def inverse_zigzag_order(string, rows, cols):
    matrix = np.zeros((rows, cols), dtype=int)
    index = -1
    bound = rows + cols - 1
    array = list(map(int, string.split()))
    for i in range(bound):
        if i % 2 == 0:
            r = i if i < rows else rows - 1
            c = 0 if i < rows else i - rows + 1
            while r >= 0 and c < cols:
                index += 1
                matrix[r, c] = array[index]
                r -= 1
                c += 1
        else:
            r = 0 if i < cols else i - cols + 1
            c = i if i < cols else cols - 1
            while r < rows and c >= 0:
                index += 1
                matrix[r, c] = array[index]
                r += 1
                c -= 1
    return matrix

#Даунсэмплинг
def downsample_image(image, Cx, Cy, method='remove'):
    N, M = image.shape
    new_N, new_M = N // Cx, M // Cy

    if method == 'remove':
        return image[::Cx, ::Cy]

    downsampled = np.zeros((new_N, new_M))

    for i in range(new_N):
        for j in range(new_M):
            block = image[i * Cx:(i + 1) * Cx, j * Cy:(j + 1) * Cy]

            if method == 'mean':
                downsampled[i, j] = np.mean(block)
            elif method == 'closest_to_mean':
                mean_val = np.mean(block)
                downsampled[i, j] = block.flat[np.abs(block - mean_val).argmin()]

    return downsampled


# image = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12],
#     [13, 14, 15, 16]
# ])
#
# Cx, Cy = 2, 2
#
# print("Remove method:")
# print(downsample_image(image, Cx, Cy, method='remove'))
#
# print("\nMean method:")
# print(downsample_image(image, Cx, Cy, method='mean'))
#
# print("\nClosest to mean method:")
# print(downsample_image(image, Cx, Cy, method='closest_to_mean'))



#Апсэмплинг
def upsample_image(image, Cx, Cy):
    N, M = image.shape
    new_N, new_M = N * Cx, M * Cy

    upsampled = np.zeros((new_N, new_M), dtype=image.dtype)

    for i in range(N):
        for j in range(M):
            upsampled[i * Cx:(i + 1) * Cx, j * Cy:(j + 1) * Cy] = image[i, j]

    return upsampled


# image = np.array([
#     [1, 2],
#     [3, 4]
# ])
#
# Cx, Cy = 2, 3
#
# print("Original image:")
# print(image)
#
# print("\nUpsampled image:")
# print(upsample_image(image, Cx, Cy))


#Прямое и обратное дискретное косинусное преобразование\
def GetH():
    h = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == 0:
                h[i][j] = np.sqrt(1 / 8)
            else:
                h[i][j] = np.sqrt(2 / 8) * np.cos(np.pi * (2 * j + 1) * i / (2 * 8))

    return h


def dct(a, h):
    return np.matmul(np.matmul(h, a), np.transpose(h))


def idct(c, h):
    return np.int32(np.matmul(np.matmul(np.transpose(h), c), h))


# block = np.array([
#     [52, 55, 61, 66, 70, 61, 64, 73],
#     [63, 59, 55, 90, 109, 85, 69, 72],
#     [62, 59, 68, 113, 144, 104, 66, 73],
#     [63, 58, 71, 122, 154, 106, 70, 69],
#     [67, 61, 68, 104, 126, 88, 68, 70],
#     [79, 65, 60, 70, 77, 68, 58, 75],
#     [85, 71, 64, 59, 55, 61, 65, 83],
#     [87, 79, 69, 68, 65, 76, 78, 94]
# ], dtype=np.int32)
#
# h = GetH()
#
# print("Original block:")
# print(block)
#
# dct_coeffs = dct(block, h)
# print("\nDCT coefficients:")
# print(dct_coeffs)
#
# reconstructed_block = idct(dct_coeffs, h)
# print("\nReconstructed block:")
# print(reconstructed_block)


# Получение матрицы квантования для заданного Q
import numpy as np
def get_quantization_matrix(Q):
    Q50 = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    if Q < 1:
        Q = 1
    elif Q > 100:
        Q = 100

    if Q < 50:
        scale = 5000 / Q
    else:
        scale = 200 - 2 * Q

    quant_matrix = np.floor((Q50 * scale + 50) / 100).astype(int)
    quant_matrix[quant_matrix < 1] = 1
    quant_matrix[quant_matrix > 255] = 255

    return quant_matrix


# quality = 90
# quant_matrix = get_quantization_matrix(quality)
# print("Quantization matrix for quality {}:".format(quality))
# print(quant_matrix)

#Квантование и деквантование

def quantize(dct_coeffs, quant_matrix):
    return np.round(dct_coeffs / quant_matrix).astype(np.int32)


def dequantize(quantized_coeffs, quant_matrix):
    return (quantized_coeffs * quant_matrix).astype(np.int32)


# dct_coeffs = np.array([
#         [1104, -6, -24, 15, -14, 0, 0, 0],
#         [-108, -8, 4, 0, 8, 0, 0, 0],
#         [8, 16, 20, 7, 1, 0, 0, 0],
#         [24, 10, 0, 9, 0, 1, 0, 0],
#         [10, 35, 0, 0, 0, 0, 0, 0],
#         [7, -11, 17, 0, 24, 0, 0, 1],
#         [15, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 1, 0, 0]
#     ], dtype=int)
#
# quant_matrix = get_quantization_matrix(50)
#
# print("Original DCT coefficients:")
# print(dct_coeffs)
#
# quantized_coeffs = quantize(dct_coeffs, quant_matrix)
# print("\nQuantized coefficients:")
# print(quantized_coeffs)
#
# dequantized_coeffs = dequantize(quantized_coeffs, quant_matrix)
# print("\nDequantized coefficients:")
# print(dequantized_coeffs)

#Прямое и обратное кодирование RLE
def rle_encode(s):

    encoded = []
    count = 1
    flag = chr(256)
    strlen = len(s)
    for i in range(1, strlen):
        if s[i] == s[i - 1]:
            count += 1
        else:
            if count == 1:
                encoded.append(s[i - 1])
            else:
                encoded.append(count)
                encoded.append(flag)
                encoded.append(s[i - 1])
            count = 1
    if count == 1:
        encoded.append(s[len(s) - 1])
    else:
        encoded.append(count)
        encoded.append(flag)
        encoded.append(s[len(s) - 1])
    res = ''
    for i in encoded:
        if (i == flag):
            res += flag

        else:
            res += str(i) + ' '
    return res


def rle_decode(string):

    decoded = ''
    flag = chr(256)
    i = 0
    strlen = len(string)
    while (i < (strlen - 1)):
        if (i != 0) and (string[i] == flag):
            r = l = 0
            while (string[i + r] != ' ') and ((i + r) < len(string) - 1):
                r += 1
            while (string[i - 2 - l] != ' ') and (i - 1 - l > 0):
                l += 1
            decoded = decoded[:(len(decoded) - 1 - l)]
            decoded += int(string[(i - 1 - l):(i - 1)]) * (string[(i + 1):(i + r)] + ' ')
            i += r + 1
        else:
            decoded += string[i]
            i += 1
    return decoded


# dct_coeffs = np.array([
#     [1104, -6, -24, 15, -14, 0, 0, 0],
#     [-108, -8, 4, 0, 8, 0, 0, 0],
#     [8, 16, 20, 7, 1, 0, 0, 0],
#     [24, 10, 0, 9, 0, 1, 0, 0],
#     [10, 35, 0, 0, 0, 0, 0, 0],
#     [7, -11, 17, 0, 24, 0, 0, 1],
#     [15, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 1, 0, 0]
# ], dtype=int)
#
# quant_matrix = get_quantization_matrix(50)
# quantized_coeffs = quantize(dct_coeffs, quant_matrix)
#
# print("Quantized coefficients:")
# print(quantized_coeffs)
#
# zigzag_sequence = zigzag_order(quantized_coeffs)
# print("\nZigzag sequence:")
# print(zigzag_sequence)
#
# rle_encoded = rle_encode(zigzag_sequence)
# print("\nRLE encoded sequence:")
# print(rle_encoded)
#
# rle_decoded = rle_decode(rle_encoded)
# print("\nRLE decoded sequence:")
# print(rle_decoded)



#Функция ДКП->квантование->деквантование->обратное
#ДКП


def process_image(img, Q):
    height, width = img.shape
    processed_img = np.zeros_like(img, dtype=np.float32)

    quant_matrix = get_quantization_matrix(Q)
    h = GetH()

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = img[i:i + 8, j:j + 8]
            dct_block = dct(block, h)
            quantized_block = quantize(dct_block, quant_matrix)
            dequantized_block = dequantize(quantized_block, quant_matrix)
            idct_block = idct(dequantized_block, h)
            processed_img[i:i + 8, j:j + 8] = idct_block

    processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)

    return processed_img


#Функция разбиения изображения на каналы
def imgtorawchanels(img):
    imgarr = np.array(img)
    R = []
    G = []
    B = []
    for line in imgarr:
        for pixel in line:
            R.append(pixel[0])
            G.append(pixel[1])
            B.append(pixel[2])
    return np.reshape(R, (len(imgarr), len(imgarr[0]))), np.reshape(G, (len(imgarr), len(imgarr[0]))), np.reshape(B,
                                                                                                                  (
                                                                                                                  len(imgarr),
                                                                                                                  len(
                                                                                                                          imgarr[
                                                                                                                              0])))


#Запись RLE в файл
def rle_to_file(encoded_data, filename):
    with open(filename, 'wb') as file:
        for item in encoded_data:
            file.write(ord(item).to_bytes(2, byteorder='big'))
    return


#Чтение RLE из файла
def rle_from_file(filename):
    encoded_data = ''
    with open(filename, 'rb') as file:
        while True:
            num = file.read(2)
            if not num:
                break
            encoded_data += chr(int.from_bytes(num, "big"))
    return encoded_data


#Компрессор

def process_compression(matrix, Q):
    height, width = matrix.shape
    processed_matrix = np.zeros_like(matrix, dtype=np.float32)

    quant_matrix = get_quantization_matrix(Q)
    h = GetH()

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = matrix[i:i + 8, j:j + 8]
            block -= 128
            dct_block = dct(block, h)
            quantized_block = quantize(dct_block, quant_matrix)
            processed_matrix[i:i + 8, j:j + 8] = quantized_block

    return processed_matrix


def compressor(filename, Cx, Cy, method='remove', Q=50):
    image = Image.open(filename)
    array = imgtoraw(image)
    y, Cb, Cr = rgbtoycbcr(array)
    with open(f'./{filename}_output/{filename}_int32', 'wb') as file:
        for item in y:
            file.write(int(item).to_bytes(2, byteorder='big'))
        for item in Cb:
            file.write(int(item).to_bytes(2, byteorder='big'))
        for item in Cr:
            file.write(int(item).to_bytes(2, byteorder='big'))
    res_ycbcr = []
    for i in range(len(y)):
        res_ycbcr.append([y[i], Cb[i], Cr[i]])
    res_ycbcr = np.array(res_ycbcr)
    res_ycbcr = np.reshape(res_ycbcr, (len(array), len(array[0]), 3))
    image_ycbcr = Image.fromarray(res_ycbcr.astype(np.uint8), mode='YCbCr')
    image_ycbcr.save(f"./{filename}_output/{filename}_ycbcr.jpeg")

    print(1)

    downsampled_y = downsample_image(np.reshape(y, (len(array), len(array[0]))), Cx, Cy, method)
    downsampled_Cb = downsample_image(np.reshape(Cb, (len(array), len(array[0]))), Cx, Cy, method)
    downsampled_Cr = downsample_image(np.reshape(Cr, (len(array), len(array[0]))), Cx, Cy, method)
    res_downsampled = []
    for i in range(len(array) // Cy * len(array[0]) // Cx):
        res_downsampled.append([np.reshape(downsampled_y, (len(array) // Cy * len(array[0]) // Cx))[i],
                                np.reshape(downsampled_Cb, (len(array) // Cy * len(array[0]) // Cx))[i],
                                np.reshape(downsampled_Cr, (len(array) // Cy * len(array[0]) // Cx))[i]])
    res_downsampled = np.array(res_downsampled)
    res_downsampled = np.reshape(res_downsampled, (len(array) // Cy, len(array[0]) // Cx, 3))
    image_downsampled = Image.fromarray(res_downsampled.astype(np.uint8), mode='YCbCr')
    image_downsampled.save(f"./{filename}_output/{filename}_downsampled.jpeg")

    print(2)

    quantized_y = process_compression(np.reshape(downsampled_y, (len(array) // Cy, len(array[0]) // Cx)), Q)
    quantized_Cb = process_compression(np.reshape(downsampled_Cb, (len(array) // Cy, len(array[0]) // Cx)), Q)
    quantized_Cr = process_compression(np.reshape(downsampled_Cr, (len(array) // Cy, len(array[0]) // Cx)), Q)

    print(3)

    zigzag_y = zigzag_order(quantized_y)
    zigzag_Cb = zigzag_order(quantized_Cb)
    zigzag_Cr = zigzag_order(quantized_Cr)

    print(4)

    rle_y = rle_encode(zigzag_y)
    rle_Cb = rle_encode(zigzag_Cb)
    rle_Cr = rle_encode(zigzag_Cr)

    rle_to_file(rle_y, f"./{filename}_output/{filename}_y_rle")
    rle_to_file(rle_Cb, f"./{filename}_output/{filename}_Cb_rle")
    rle_to_file(rle_Cr, f"./{filename}_output/{filename}_Cr_rle")





#Декомпрессор
def process_decompression(matrix, Q):
    height, width = matrix.shape
    processed_matrix = np.zeros_like(matrix, dtype=np.float32)

    quant_matrix = get_quantization_matrix(Q)
    h = GetH()

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = matrix[i:i + 8, j:j + 8]
            dequantized_block = dequantize(block, quant_matrix)
            idct_block = idct(dequantized_block, h)
            idct_block += 128
            idct_block[idct_block < 0] = 0
            idct_block[idct_block > 255] = 255
            processed_matrix[i:i + 8, j:j + 8] = idct_block

    return processed_matrix


def decompressor(filename, filename_y, filename_Cb, filename_Cr, height, weight, Q, Cx, Cy):
    rle_y = rle_from_file(filename_y)
    rle_Cb = rle_from_file(filename_Cb)
    rle_Cr = rle_from_file(filename_Cr)

    rle_y_decoded = rle_decode(rle_y)
    rle_Cb_decoded = rle_decode(rle_Cb)
    rle_Cr_decoded = rle_decode(rle_Cr)

    y = inverse_zigzag_order(rle_y_decoded, height, weight)
    Cb = inverse_zigzag_order(rle_Cb_decoded, height, weight)
    Cr = inverse_zigzag_order(rle_Cr_decoded, height, weight)

    y_idct = process_decompression(np.reshape(y, (height, weight)), Q)
    Cb_idct = process_decompression(np.reshape(Cb, (height, weight)), Q)
    Cr_idct = process_decompression(np.reshape(Cr, (height, weight)), Q)

    y_upsampled = upsample_image(y_idct, Cx, Cy)
    Cb_upsampled = upsample_image(Cb_idct, Cx, Cy)
    Cr_upsampled = upsample_image(Cr_idct, Cx, Cy)
    res_upsampled = []
    for i in range(height * Cy * weight * Cx):
        res_upsampled.append([np.reshape(y_upsampled, height * Cy * weight * Cx)[i],
                              np.reshape(Cb_upsampled, height * Cy * weight * Cx)[i],
                              np.reshape(Cr_upsampled, height * Cy * weight * Cx)[i]])
    res_upsampled = np.array(res_upsampled)
    height *= Cy
    weight *= Cx
    res_upsampled = np.reshape(res_upsampled, (height, weight, 3))
    image_upsampled = Image.fromarray(res_upsampled.astype(np.uint8), mode='YCbCr')
    image_upsampled.save(f"./{filename}_output/{filename}_upsampled.jpeg")

    rgb = ycbcrtorgb(res_upsampled)
    image_rgb = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
    image_rgb.save(f"./{filename}_output/{filename}_rgb.jpeg")

# img = Image.open("color.jpeg")
# img = img.resize((400, 400))
# img_array = np.array(img)
# R, G, B = imgtorawchanels(img)
# for Q in [90, 70, 50, 30, 10]:
#     processed_R = process_image(R, Q)
#     processed_G = process_image(G, Q)
#     processed_B = process_image(B, Q)
#
#     res_rgb = []
#     rpR = np.reshape(processed_R, len(img_array) * len(img_array[0]))
#     rpG = np.reshape(processed_G, len(img_array) * len(img_array[0]))
#     rpB = np.reshape(processed_B, len(img_array) * len(img_array[0]))
#     for j in range(len(rpR)):
#         res_rgb.append([rpR[j], rpG[j], rpB[j]])
#     res_rgb = np.array(res_rgb)
#     res_rgb = np.reshape(res_rgb, (len(img_array), len(img_array[0]), 3))
#     immg = Image.fromarray(res_rgb.astype(np.uint8))
#
#     immg.show()
#     immg.save(f"./quality/quality{Q}.jpeg")
#     immg.close()


filename = 'pompeii'

# img = Image.open(f"{filename}.jpeg")
# img = img.resize((400, 400))
# arr = imgtoraw(img)
# y, Cb, Cr = rgbtoycbcr(arr)
# y1 = np.reshape(y, (len(arr), len(arr[0])))
# Cb1 = np.reshape(Cb, (len(arr), len(arr[0])))
# Cr1 = np.reshape(Cr, (len(arr), len(arr[0])))
# imy = Image.fromarray(y1.astype(np.uint8))
# imcb = Image.fromarray(Cb1.astype(np.uint8))
# imcr = Image.fromarray(Cr1.astype(np.uint8))
# res_ycbcr = []
# for i in range(len(y)):
#     res_ycbcr.append([y[i], Cb[i], Cr[i]])
# res_ycbcr = np.array(res_ycbcr)
# res_ycbcr = np.reshape(res_ycbcr, (len(arr), len(arr[0]), 3))
# immg = Image.fromarray(res_ycbcr.astype(np.uint8), mode='YCbCr')
#
# img.show()
# #value=input("Enter to continue:")
# imy.show()
# #value=input("Enter to continue:")
# imcb.show()
# #value=input("Enter to continue:")
# imcr.show()
# #value=input("Enter to continue:")
# immg.show()
# #value=input("Enter to continue:")
#
# img.close()
# imy.close()
# imcb.close()
# imcr.close()
# immg.close()
#
# rgb = ycbcrtorgb(res_ycbcr)
# image_rgb = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
# image_rgb.show()





compressor(f"{filename}.jpeg", 2, 2, Q=75)

decompressor(f"{filename}.jpeg", f"./{filename}.jpeg_output/{filename}.jpeg_y_rle",
                 f"./{filename}.jpeg_output/{filename}.jpeg_Cb_rle", f"./{filename}.jpeg_output/{filename}.jpeg_Cr_rle",
                 200, 200, 75, 2, 2)


# image = Image.open('lena.jpeg')
# new_image = image.resize((400,400))
# new_image.save('lena.jpeg')
