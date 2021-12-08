# Customer-Loyalty-Modelling
This project aims to model and predict whether a customer will leave the bank in the near future. **The detailed analysis and insights are shown in the Jupyter Notebook.**
The management team is able to use the analysis and insights for their decision making to enhance customer relationship and retain the customers as much as possible.

A web-based application was also developed based on the best performing model and Flask framework. Once opened from browser,  the below page will show up:

![Alt text](https://github.com/jsun66/Customer-Loyalty-Modelling/blob/main/web%20page.JPG)

Users can input all the numerical values and select the appropriated categorical values from the drop-down menus then click the "Predict" button, a "Yes" or "No" will show up after "Will Exit?:". Two successfully predicted examples (one is "will exit" and the other one is "will not exit") are given below:

![Alt text](https://github.com/jsun66/Customer-Loyalty-Modelling/blob/main/Example-Will%20Exit.JPG)

![Alt text](https://github.com/jsun66/Customer-Loyalty-Modelling/blob/main/Example-Will%20Not%20Exit.JPG)

Different binary classification models were implemented and XGBoost gives the best result. The classification Report of the XGBoost Model is shown below:

![Alt text](https://github.com/jsun66/Customer-Loyalty-Modelling/blob/main/Classification%20report%20of%20XGBoost%20Mode.JPG)

The 5-fold Cross Validation Results of f1-Score is shown below:

![Alt text](https://github.com/jsun66/Customer-Loyalty-Modelling/blob/main/5-fold%20Cross%20Validation%20Result%20of%20F1-Score.JPG)
