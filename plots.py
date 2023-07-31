import shap
import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt

base_path = "C:/Users/clalk/JLUbox/Transkriptanalysen/2 TOPIC MODELING/Analysen/"
sub_folder_processing = "data/processing"
sub_folder_transkripte = "data/transkripte"
sub_folder_Patient = "data/Patient_classed/hscl_diff"
sub_folder_output = "data/Patient_classed/hscl_diff" # oder "srs" oder "hscl_nächste_sitzung


path = os.path.join(base_path, sub_folder_output)
os.chdir(path)
print(path)



df_sh = pd.read_excel('sh_values.xlsx', index_col=0)
df_bs = pd.read_excel('bs_values.xlsx', index_col=0)
df_sh_data = pd.read_excel('sh_data.xlsx', index_col=0)
importance = pd.read_excel('Feature_list.xlsx')


sh_values = df_sh.values
bs_values = df_bs.values
sh_data = df_sh_data.values
feature_list = importance["Feature"].tolist()

shap_values2 = shap.Explanation(values=sh_values,
                                   base_values=bs_values, data=sh_data,
                                   feature_names=feature_list)
shap.summary_plot(shap_values2, plot_size=(14, 5), max_display=7, show=True)
shap.summary_plot(shap_values2, plot_size=(14, 5), max_display=7, show=False)
plt.savefig('top13.png')

