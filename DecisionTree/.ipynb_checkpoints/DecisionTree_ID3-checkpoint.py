# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Decision Tree算法：ID3 

# ## 画图 tree_plotter.py

# +
import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', \
                             xytext=center_pt, textcoords='axes fraction', \
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            thisDepth = get_tree_depth(second_dict[key]) + 1
        else:
            thisDepth = 1
        if thisDepth > max_depth:
            max_depth = thisDepth
    return max_depth


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


# -

# ## 导入包 

# +
# -*- coding: utf-8 -*-
from math import log
import operator

def create_data_set():
    """
    创建样本数据
    :return:
    """
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_shannon_ent(data_set):
    """
    计算信息熵
    :param data_set: 如： [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    :return:
    """
    num = len(data_set)  # n rows
    # 为所有的分类类目创建字典
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]  # 取得最后一列数据
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    # 计算香浓熵
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num
        shannon_ent = shannon_ent - prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    返回特征值等于value的子数据集，切该数据集不包含列（特征）axis
    :param data_set:  待划分的数据集
    :param axis: 特征索引
    :param value: 分类值
    :return:
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    按照最大信息增益划分数据
    :param data_set: 样本数据，如： [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    :return:
    """
    num_feature = len(data_set[0]) - 1 # 特征个数，如：不浮出水面是否可以生存	和是否有脚蹼
    base_entropy = calc_shannon_ent(data_set) # 经验熵H(D)
    best_info_gain = 0
    best_feature_idx = -1
    for feature_idx in range(num_feature):
        feature_val_list = [number[feature_idx] for number in data_set]  # 得到某个特征下所有值（某列）
        unique_feature_val_list = set(feature_val_list)  # 获取无重复的属性特征值
        new_entropy = 0
        for feature_val in unique_feature_val_list:
            sub_data_set = split_data_set(data_set, feature_idx, feature_val)
            prob = len(sub_data_set) / float(len(data_set)) # 即p(t)
            new_entropy += prob * calc_shannon_ent(sub_data_set) #对各子集香农熵求和
        info_gain = base_entropy - new_entropy # 计算信息增益，g(D,A)=H(D)-H(D|A)
        # 最大信息增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_idx = feature_idx

    return best_feature_idx


def majority_cnt(class_list):
    """
    统计每个类别出现的次数，并按大到小排序，返回出现次数最大的类别标签
    :param class_list: 类数组
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reversed=True)
    print(sorted_class_count[0][0])
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    构建决策树
    :param data_set: 数据集合，如： [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    :param labels: 标签数组，如：['no surfacing', 'flippers']
    :return:
    """
    class_list = [sample[-1] for sample in data_set] # ['yes', 'yes', 'no', 'no', 'no']
    # 类别相同，停止划分
    if class_list.count(class_list[-1]) == len(class_list):
        return class_list[-1]
    # 长度为1，返回出现次数最多的类别
    if len(class_list[0]) == 1:
        return majority_cnt((class_list))
    # 按照信息增益最高选取分类特征属性
    best_feature_idx = choose_best_feature_to_split(data_set)  # 返回分类的特征的数组索引
    best_feat_label = labels[best_feature_idx]  # 该特征的label
    my_tree = {best_feat_label: {}}  # 构建树的字典
    del (labels[best_feature_idx])  # 从labels的list中删除该label，相当于待划分的子标签集
    feature_values = [example[best_feature_idx] for example in data_set]
    unique_feature_values = set(feature_values)
    for feature_value in unique_feature_values:
        sub_labels = labels[:]  # 子集合
        # 构建数据的子集合，并进行递归
        sub_data_set = split_data_set(data_set, best_feature_idx, feature_value) # 待划分的子数据集
        my_tree[best_feat_label][feature_value] = create_tree(sub_data_set, sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """
    决策树分类
    :param input_tree: 决策树
    :param feat_labels: 特征标签
    :param test_vec: 测试的数据
    :return:
    """
    first_str = list(input_tree.keys())[0]  # 获取树的第一特征属性
    second_dict = input_tree[first_str]  # 树的分子，子集合Dict
    feat_index = feat_labels.index(first_str)  # 获取决策树第一层在feat_labels中的位置
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
            return class_label


data_set, labels = create_data_set()
decision_tree = create_tree(data_set, labels)
print ("决策树：", decision_tree)
data_set, labels = create_data_set()
print ("（1）不浮出水面可以生存，无脚蹼：", classify(decision_tree, labels, [1, 0]))
print ("（2）不浮出水面可以生存，有脚蹼：", classify(decision_tree, labels, [1, 1]))
print ("（3）不浮出水面可以不能生存，无脚蹼：", classify(decision_tree, labels, [0, 0]))
create_plot(decision_tree)
