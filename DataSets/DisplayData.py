import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DataSets.ImportData import Datasets

class DisplayData:
    def __init__(self, data_panda):
        self.data_panda = data_panda

    def display_Data_Quantity(self):

        plt.figure(figsize=(20, 8))
        sns.countplot(data=self.data_panda, x='quality')
        plt.show()

    def display_Data_Pairwise(self):
        sns.set_style("dark")
        plt.figure(figsize=(10, 8))
        colnm = df.columns.tolist()[:11] + ['quality']
        mcorr = df[colnm].corr()  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
        mask = np.zeros_like(mcorr, dtype=np.bool_)  # 创建一个mcorr一样的全False矩阵
        cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 建立一个发散调色板
        g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
        print("\nFigure: Pairwise Correlation Plot")
        plt.show()

    def display_Data_Quality(self):
        sns.set_style('ticks')  # 设置图表主题背景为十字叉
        sns.set_context("notebook", font_scale=1.1)  # 设置图表样式

        colnm = df.columns.tolist()[:11]
        plt.figure(figsize=(10, 8))

        for i in range(11):
            plt.subplot(4, 3, i + 1)
            sns.boxplot(x='quality', y=colnm[i], data=df, width=0.6)
            plt.ylabel(colnm[i], fontsize=12)
        plt.tight_layout()
        print("\nFigure: Physicochemical Properties and Wine Quality by Boxplot")
        plt.show()


if __name__ == '__main__':


    location_red = "..\\DataSets\\winequality-red.csv"
    datasets_red = Datasets(location_red)
    datasets_red.loadData()
    df = datasets_red.displayData()

    display = DisplayData(df)
    #display.display_Data_Quantity()
    #display.display_Data_Pairwise()
    display.display_Data_Quality()



