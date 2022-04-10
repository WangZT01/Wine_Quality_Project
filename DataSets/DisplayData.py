import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DataSets.ImportData import Datasets


'''

This class is used to visualize data.
'''
class DisplayData:
    def __init__(self, data_panda):
        self.data_panda = data_panda


    '''
    Output the number of data.
    '''
    def display_Data_Quantity(self):

        plt.figure(figsize=(20, 8))
        ax = sns.countplot(data=self.data_panda, x='quality')
        for p in ax.patches:
            ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01))
        plt.show()

    '''
    Output the Pearson coefficient of features.
    '''
    def display_Data_Pairwise(self, df):
        sns.set_style("dark")
        plt.figure(figsize=(10, 8))
        colnm = df.columns.tolist()[:11] + ['quality']
        mcorr = df[colnm].corr()
        mask = np.zeros_like(mcorr, dtype=np.bool_)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
        print("\nFigure: Pairwise Correlation Plot")
        plt.savefig('../Images/Pairwise.jpg')
        plt.show()

    '''
    Output the relation between quality and features.
    '''
    def display_Data_Quality(self, df):
        sns.set_style('ticks')
        sns.set_context("notebook", font_scale=1.1)

        colnm = df.columns.tolist()[:11]

        plt.figure(figsize = (10,8))
        for i in range(11):
            plt.subplot(4, 3, i + 1)
            #plt.figure(figsize=(10, 8))
            sns.boxplot(x='quality', y=colnm[i], data=df, width=0.6)
            plt.ylabel(colnm[i], fontsize=12)
            plt.savefig( '../Images/wine_qualiy_' + colnm[i] +'.jpg')
        plt.tight_layout()
        print("\nFigure: Physicochemical Properties and Wine Quality by Boxplot")
        plt.show()

if __name__ == '__main__':


    location_red = "..\\DataSets\\winequality-red.csv"
    datasets_red = Datasets(location_red)
    datasets_red.loadData()
    df = datasets_red.displayData()

    display = DisplayData(df)
    display.display_Data_Quantity()
    display.display_Data_Pairwise(df)
    display.display_Data_Quality()


    location_white = "..\\DataSets\\winequality-white.csv"
    datasets_white = Datasets(location_white)
    datasets_white.loadData()
    df_white = datasets_white.displayData()

    display_white = DisplayData(df_white)
    #display_white.display_Data_Quantity()
    display_white.display_Data_Pairwise(df_white)
    display_white.display_Data_Quality(df_white)
    #display_white.dispaly_radviz(df_white)

