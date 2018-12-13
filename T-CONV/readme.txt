1.首先将数据处理成hdf5格式(储存训练集验证集和测试集)，处理后的字段可以参照./src/Data/hdf5_dataset_cut.py。
2.使用./src/Data/preprocess/下的cluster_arrival.py对目的点聚类，其中的超参数为bw,会影响最终性能，可根据地图大小和目的点密集程度自行选择。
3.在Config中__init__.py中设置聚类后的pkl文件位置，Config中__init__.py中的data.path在Data目录下的__init__.py设置。
4.Data中__init__.py的train_gps_mean是地图的中心坐标，train_gps_std是中心距离上下和两侧的距离。
5.Data中Load_dataset_le中self.size代表训练集大小，自行设置，主要用于从和hdf5中加载数据集。
6.Data中transform2_le.py第175行替换掉成训练集的大小，self.level2_lat和self.level2_lon设置想把地图设置成多少个网格。valid_transform2_le.py也需要设置self.level2_lat和self.level2_lon。
7.Models中conn-local.py第188行需要设置数据位置。




