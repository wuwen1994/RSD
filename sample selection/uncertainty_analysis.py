import cv2 as cv
import glob
import numpy as np
import math

sample_paths = glob.glob(r"./sample/*.jpg")
for i, sample_path in enumerate(sample_paths):
    file_name = sample_path.split("\\")[-1].split(".")[0]
    predictions_path = glob.glob(r"./results/" + file_name + "*.png")
    # 该列表为二维列表，每一行代表每个precition在每一点上的预测概率
    history_prediction = []

    for prediction_path in predictions_path:
        current_list = []
        predcition = cv.imread(prediction_path, cv.IMREAD_GRAYSCALE)
        height, width = predcition.shape[0:]
        for h in range(height):
            for w in range(width):
                current_list.append(predcition[h, w] / 255)
        history_prediction.append(current_list)

    # 转化为array并转置为 65536行10列, 设置分裂阈值,得到历史预测分类值
    t = 0.3
    history_array = np.array(history_prediction).T
    history_array[history_array >= t] = 1
    history_array[history_array < t] = 0
    number_of_samples, len_of_history = history_array.shape[0:]

    # 通过香农熵分析每个像素的类别概率和最常出现的标签
    P = []  # P(c|x;w)
    label = []  # cn
    for n in range(number_of_samples):
        number_of_non_shadow = 0
        number_of_shadow = 0
        pro = []
        for l in range(len_of_history):
            if history_array[n, l] == 0:
                number_of_non_shadow += 1
            else:
                number_of_shadow += 1
        pro_shadow = number_of_shadow / len_of_history
        pro_non_shadow = number_of_non_shadow / len_of_history
        P.append(pro_shadow)
        if pro_shadow >= pro_non_shadow:
            label.append(1)
        else:
            label.append(0)

    # 将P计算为置信度，F = 1-entropy(P(c|x;w))
    F = []
    noise = 0.0000001  # 防止被除数为0，无法计算香农熵
    for p in P:
        p = abs(p - noise)
        f = 1 - (-1 * p * math.log(p, 2) - (1 - p) * math.log((1 - p), 2))
        F.append(f)

    # 通过阈值筛选置信度低的像素，记住其下标，z在label列表中将其辅助为0.5
    theta = 0.531  # 约有8成以上的预测概率才视为可靠标签
    for index, f in enumerate(F):
        if f <= theta:
            label[index] = 0.5

    shadow = 0
    non_shadow = 0
    noisy_label = 0
    for l in label:
        if l == 0:
            non_shadow += 1
        elif l == 1:
            shadow += 1
        else:
            noisy_label += 1
    noisy_rate = round(noisy_label / 65536, 2)
    print("%s-th image's noisy label probability: %s" % (str(i + 1), str(noisy_rate)))

    # 可视化噪声标签
    label_array = np.array(label).reshape((height, width))
    image = np.ndarray((height, width, 3), dtype="uint8")

    for h in range(height):
        for w in range(width):
            if label_array[h, w] == 0:
                image[h, w, 0] = 0
                image[h, w, 1] = 0
                image[h, w, 2] = 0
            elif label_array[h, w] == 1:
                image[h, w, 0] = 255
                image[h, w, 1] = 255
                image[h, w, 2] = 255
            elif label_array[h, w] == 0.5:
                image[h, w, 0] = 0
                image[h, w, 1] = 0
                image[h, w, 2] = 255

    cv.imwrite("./results/" + file_name + "_Noisy_labels.png", image)
    # 记录每张图片的noise sample比例，绘制直方图
    with open("noisy_rate_ISTD.txt", "a") as f:
        line_content = file_name + '\t' + str(noisy_rate) + "\n"
        f.write(line_content)
