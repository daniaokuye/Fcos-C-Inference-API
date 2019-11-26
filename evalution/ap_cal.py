import matplotlib.pyplot as plt
import numpy as np


def precision_recall_curve(y_true, y_score, TPN, pos_label=None):
    if pos_label is None:
        pos_label = 1
    # 不同的排序方式，其结果也会有略微差别，
    # 比如 kind="mergesort" 的结果跟kind="quicksort"的结果是不同的，
    # 这是因为归并排序是稳定的，快速排序是不稳定的，sklearn中使用的是 kind="mergesort"
    desc_score_indices = np.argsort(y_score, kind="quicksort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # 确定阈值下标索引，score中可能会有重复的分数，在sklearn中，重复的元素被去掉了
    # 本来以为去除重复分数会影响结果呢，因为如果两个样本有重复的分数，一个标签是1，一个是0，
    # 感觉去掉哪个都会影响结果啊，但是实验发现竟然不影响结果，有点纳闷，以后有时间再分析吧
    # distinct_value_indices = np.where(np.diff(y_score))[0]
    # threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # 这里没有去除重复分数
    threshold_idxs = np.arange(y_score.size)
    # 按照阈值依次降低的顺序，确定当前阈值下的true positives 个数，tps[-1]对应于所有的正例数量
    tps = np.cumsum(y_true * 1.)[threshold_idxs]
    # 计算当前阈值下的 false positives 个数，
    # 它与 tps的关系为fps=1+threshold_idxs-tps，这个关系是比较明显的
    fps = 1 + threshold_idxs - tps
    # y_score[threshold_idxs]把对应下标的阈值取出
    thresholds = y_score[threshold_idxs]
    precision = tps / (tps + fps)
    recall = tps / TPN
    # 这里与sklearn有略微不同，即样本点全部输出，令last_ind = tps.size，即可
    last_ind = tps.size
    sl = slice(0, last_ind)
    return np.r_[1, precision[sl]], np.r_[0, recall[sl]], thresholds[sl]


def average_precision_approximated(y_true, y_predict, TPN):
    """
    计算approximated形式的ap，每个样本点的分数都是recall的一个cutoff
    :param y_true: 标签
    :param y_predict: 实际预测得分
    :return: precision，recall，threshold，average precision
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_predict, TPN, pos_label=1)
    average_precision = np.sum(np.diff(recall) * np.array(precision)[1:])
    return precision, recall, thresholds, average_precision


def average_precision_interpolated(y_true, y_predict, TPN):
    """
    计算interpolated形式的ap，每个正样本对应的分数都是recalll的一个cutoff
    :param y_true: 标签
    :param y_predict: 实际预测得分
    :return: precision，recall，threshold，average precision
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_predict, TPN, pos_label=1)
    # 获取recall轴上的分割，np.insert(recall, 0 , -1, axis=0)是为了保证获取到重复recall的第一个索引值
    # 因为重复的recall中只有对应的第一个precision是最大的，我们只需要获取这个最大的precision
    # 或者说每遇到一个正样本，需要将其对应的recall值作为横轴的切分
    recall_cutoff_index = np.where(
        np.diff(np.insert(recall, 0, -1, axis=0)))[0]
    # 从recall的cutoff 索引值开始往后获取precision最大值，相同的precision只取索引值最大的那个
    # P(r) = max{P(r')} | r'>=r
    precision_cutoff_index = []
    for index in recall_cutoff_index:
        precision_cutoff_index.append(
            max([x for x in np.where(precision == np.max(precision[index:]))[0] if x >= index]))
    # interpolated_idx=np.unique(interpolated_cutoff)
    # 从原始的precision和recall中截取对应索引的片段，即可得到 interpolated 方式下的precision，recall以及AP
    precision_interpolated = precision[precision_cutoff_index]
    recall_interpolated = recall[recall_cutoff_index]
    # 以上获得的 recall_cutoff_index 和 precision_cutoff_index 切片包含人为添加的0 和 1（为了画图时与坐标轴封闭）
    # 而计算thresholds_interpolated时要去掉相应索引值的影响
    # 阈值不包括recall=0
    thresholds_interpolated = thresholds[
        [x - 1 for x in recall_cutoff_index if 0 <= x - 1 < thresholds.size]]
    # 按说ap计算应该按照面积的方式计算，也就是下面被注释的部分，但论文里面是直接计算均值，
    # 这里也直接计算均值，因为阈值不包括recall=0，所以这种情况下二者结果是一样的
    # average_precision = np.sum(precision_interpolated[1:]) / 11
    average_precision = np.sum(
        np.diff(recall_interpolated) * np.array(precision_interpolated)[1:])
    return precision_interpolated, recall_interpolated, thresholds_interpolated, average_precision


def average_precision_11point_interpolated(y_true, y_predict, TPN):
    """
    计算 11point形式的 ap
    :param y_true: 标签
    :param y_predict: 实际预测得分
    :return: precision，recall，threshold，average precision
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_predict, TPN, pos_label=1)
    recall_11point_cutoff = np.array(
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # 计算离11个cutoff最近的样本点
    recall_cutoff_index = []
    for cutoff in recall_11point_cutoff:
        try:
            recall_cutoff_index.append(np.where(recall >= cutoff)[0][0])
        except Exception as e:
            pass
    precision_cutoff_index = []
    for index in recall_cutoff_index:
        precision_cutoff_index.append(
            max([x for x in np.where(precision == np.max(precision[index:]))[0] if x >= index]))
    precision_11point = precision[precision_cutoff_index]
    recall_11point = recall[recall_cutoff_index]
    # 此处阈值包括recall=0，因为是11points
    thresholds_11point = thresholds[
        [x - 1 for x in recall_cutoff_index if -1 <= x - 1 < thresholds.size]]
    # 此处阈值包括recall=0，因为是11points，所以这种情况下两种计算AP的方式结果不同，有略微差别
    pr = np.zeros_like(recall_11point_cutoff)
    pr[:len(precision_11point)] = precision_11point
    # average_precision = np.sum(precision_11point) / 11

    average_precision = np.sum(
        0.1 * np.array(precision_11point)[1:])
    # 此处直接返回 recall_11point_cutoff，实际上返回 recall_11point 也是可以的，
    # 差别就是图线的转折点不在[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]这11个刻度处
    # return precision_11point, recall_11point, thresholds_11point, average_precision
    return pr, recall_11point_cutoff, thresholds_11point, average_precision


def plot_main(y_test, y_score, TPN, filename):
    # 计算AP，并画图
    precision_approximated, recall_approximated, _, ap_approximated = \
        average_precision_approximated(y_test, y_score, TPN)
    precision_interpolated, recall_interpolated, _, ap_interpolated = \
        average_precision_interpolated(y_test, y_score, TPN)
    precision_11point, recall_11point, _, ap_11point = \
        average_precision_11point_interpolated(y_test, y_score, TPN)
    print('Approximated average precision-recall score: {0:0.5f}'.format(
        ap_approximated))
    print('Interpolated average precision-recall score: {0:0.5f}'.format(
        ap_interpolated))
    print('Interpolated at fixed 11 points average precision-recall score: {0:0.5f}'.format(
        ap_11point))

    # print the AP plot
    fig1 = plt.figure('fig1')
    # plt.subplot(311)
    plt.plot(recall_approximated, precision_approximated,
             color='r', marker='o', mec='m', ms=3)
    plt.step(recall_approximated, precision_approximated,
             color='c', where='pre')
    plt.fill_between(recall_approximated, precision_approximated, step='pre', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1])
    plt.title('2-class Precision-Recall curve(Approximated): AP={0:0.5f}'.format(
        ap_approximated))
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    plt.legend(('PR-curve', 'Approximated-PR-curve', 'Approximated-AP'),
               loc='upper right')
    fig2 = plt.figure('fig2')
    # plt.subplot(312)
    plt.plot(recall_approximated, precision_approximated,
             color='r', marker='o', mec='m', ms=3)
    plt.plot(recall_interpolated, precision_interpolated,
             color='c', marker='o', mec='g', ms=3, alpha=0.5)
    plt.fill_between(recall_interpolated, precision_interpolated, step='pre', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1])
    plt.title('2-class Precision-Recall curve(Interpolated): AP={0:0.5f}'.format(
        ap_interpolated))
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    plt.legend(('PR-curve', 'Interpolated-PR-curve', 'Interpolated-AP'),
               loc='upper right')
    fig3 = plt.figure('fig3')
    # plt.subplot(313)
    plt.plot(recall_approximated, precision_approximated,
             color='r', marker='o', mec='m', ms=3)
    plt.plot(recall_11point, precision_11point,
             color='c', marker='o', mec='g', ms=3)
    plt.fill_between(recall_11point, precision_11point, step='pre', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1])
    plt.title('2-class Precision-Recall curve(Interpolated_11point): AP={0:0.5f}'.format(
        ap_11point))
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    plt.legend(('PR-curve', '11point-PR-curve', '11point-AP'),
               loc='upper right')
    # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
    #                     wspace=0.35)
    # plt.savefig(filename)
    # plt.show()
