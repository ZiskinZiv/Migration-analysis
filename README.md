## Migration networks and housing prices analysis and ML tools
This repository is used to model and analyze the housing prices in Israel (1998-2021) and to study the migration networks in Israel (2000-2017)
For now, we have one project almost ready to be published:
### Two decades of housing prices in Israel: A Machine Learning approach

#### Project description
In this project we analyse and model the apartment prices of ~1 million transactions in Israel for the years 2000-2019. We use Multiple Linear Regression (MLR) with only two hedonic features (number of rooms and whether the apartment is new or not). Since the transactions have geographical coordinates, we add predictors that have geospatial features such as distance to employment centers, socio-economic status, etc. Furthermore, we use a more complex ML model (RF) and employ Explainable AI methods to gain insights from the results.

#### Example of the number of rooms and apartment status on long term prices using Multiple Linear Regression (MLR)
![](Figures/Nadlan_MLR_rooms_new_year.png?raw=true "mlr")
The rise in prices for 3 (blue), 4 (orange) and 5 (green) rooms apartments in Israel for the years 2000-2019. The apartment status, i.e., whether it is new (dashed x) or not (solid circle) is indicted as well. The shaded areas represent the model uncertainty (95% CIs).

#### Example of feature importances using Explainable AI methods on the RF model
![](Figures/Nadlan_Shap_values.png?raw=true "shap")
Mean SHAP values for each of the six predictors where the color of each predictor's values is indicated in the colorbar. The effect on the model is measured on the x-axis. For example, the New status of an apartment which is represented by a red color has a positive affect on the price while a not new apartment (blue) lowers the price. The distribution of the samples can also be insightful (here we used 1000 randomly samples).



### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
### Authors

* **Shlomi Ziskin Ziv** - *shlomiziskin@gmail.com*


