import pandas as pd
import os
import csv
import sys
import open3d as o3

from pandas.io.parsers import read_csv

data = os.listdir("datasets_lidar/chair")  # get chair folder
# print(data)

dfs = []

# print(sys.path)
source = o3.geometry.PointCloud()
print(source)

# for i in range(4):
#     print(i)

# for file_name in data:
#     # print(file_name)
#     # ファイル名がcsvなら読み込む
#     if file_name == "chair_1921680100.csv":
#         # if file_name.endswith('.csv'):
#         print(file_name)
#         with open("datasets_lidar/chair/" + file_name) as f:
#             print(f.read())
#             # print(a)

# dfs.append()

# df = pd.concat(dfs)
# print(df)
