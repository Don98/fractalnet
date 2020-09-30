# coding=utf8
from __future__ import absolute_import, division, print_function
import h5py

if __name__ == '__main__':
    f = h5py.File('../fractalnet-34.caffemodel.h5', 'r')
    for group_name in f.keys():
        # print(group)
        # 根据一级组名获得其下面的组
        group = f[group_name]
        for sub_group_name in group.keys():
            # print('----'+subgroup)
            # 根据一级组和二级组名获取其下面的dataset
            dataset = f[group_name + '/' + sub_group_name]
            # 遍历该子组下所有的dataset
            for dset in dataset.keys():
                # 获取dataset数据
                sub_dataset = f[group_name + '/' + sub_group_name + '/' + dset]
                data = sub_dataset[()]
                print(sub_dataset.name, data.shape)
