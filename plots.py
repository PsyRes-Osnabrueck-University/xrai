import shap
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_path = "C:/Users/clalk/JLUbox/Transkriptanalysen/2 TOPIC MODELING/Analysen/"

sub_folder_output = "data_mixed/Patient/Allianz"# oder "srs" oder "hscl_nächste_sitzung


path = os.path.join(base_path, sub_folder_output)
os.chdir(path)
print(path)

filename = "OUT.xlsx"
df_sh_values = pd.read_excel(filename, sheet_name="sh_values")
df_bs_values = pd.read_excel(filename, sheet_name="bs_values")
df_sh_data = pd.read_excel(filename, sheet_name="sh_data")
feature_list = pd.read_excel("SHAP-IMPORTANCE.xlsx")["Feature"].tolist()

sh_values = df_sh_values.values
bs_values = np.reshape(df_bs_values.values, (-1,))
sh_data = df_sh_data.values

shap_values2 = shap.Explanation(values=sh_values,
                                   base_values=bs_values, data=sh_data,
                                   feature_names=feature_list)
shap.summary_plot(shap_values2, plot_size=(14, 6), max_display=9, show=True)
shap.summary_plot(shap_values2, plot_size=(14, 6), max_display=9, show=False)
plt.savefig('top13.png')

shap.plots.waterfall(shap_values2[50], max_display=20, show=False)
plt.gcf().set_size_inches(50, 15)
plt.savefig('waterfall_plot.png')
plt.show()