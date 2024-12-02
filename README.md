## Project descriptions
**XRAI (eXplanable Regression-based Artificial Intelligence)** is a package for the integration of regression-based machine-learning and eXplainable AI via the SHAP-package. XRAI allows the prediction of a target variable via several ML algorithms from scikit-learn (Lasso, Elastic Net, Random Forest, Support Vector Regression, XGBoost, and a Support Vector Regression meta-learner). Algorithm selection is conducted via nested cross-validation.

Specifications can be selected regarding the internal and external folds, the selected algorithms and the use of feature selection via featureWiz. The cross-validation scheme can be safed for reproductibility. SHAP allows the estimation of feature importance both for individual predictions, as well as across the whole dataset.   

## Usage
This package requires a dataset that contains ....

```python
from xrai import Transform
from xrai import Preparation

# file_path = "path/to/your/excel/file.xlsx"

preparation = Preparation(base_path=os.getcwd(),
                                          file_name="distortions_final.xlsx",
                                          outcome=None,
                                          outcome_list=[],
                                          classed_splits=False,
                                          outcome_to_features=[],
                                          test_sets=10,
                                          val_sets=5)
transformer = Transform(dataPrepared=preparation)
transformer.gen_plots()
```

This produces plots as follows: 
### SHAP summary plot
![SHAP summary plot](https://github.com/PsyRes-Osnabrueck-University/xrai/blob/fdd2e1ed3634170229fc6124847c115ce4c6f063/images/heatmap.png)

![SHAP waterfall plot](https://github.com/PsyRes-Osnabrueck-University/xrai/blob/fdd2e1ed3634170229fc6124847c115ce4c6f063/images/waterfall_plot.png)

![SHAP Heatmap plot](https://github.com/PsyRes-Osnabrueck-University/xrai/blob/fdd2e1ed3634170229fc6124847c115ce4c6f063/images/heatmap.png)