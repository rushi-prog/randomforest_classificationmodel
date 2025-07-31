# randomforest_classificationmodel
This project aims to predict whether a customer will accept a holiday package offer using a Random Forest Classifier. The dataset includes various features such as age, gender, occupation, passport ownership, frequent flyer status, and preferred property star rating. The goal is to help travel companies better target potential customers based on their profile and past travel behavior.

We began with a thorough exploratory data analysis (EDA), identifying missing values, imbalanced classes, and key patterns in the data. Preprocessing was done using pipelines — categorical features were handled with OneHotEncoding, numerical features were scaled using StandardScaler, and everything was combined using a ColumnTransformer. We used a stratified train-test split to maintain the balance of the target classes.

For modeling, we used a RandomForestClassifier with class weights set to 'balanced' to address class imbalance. We then performed hyperparameter tuning using GridSearchCV to optimize model performance. The model was evaluated using accuracy, F1 score, classification report, confusion matrix, and ROC AUC curve. These metrics ensured both performance and fairness in prediction.

One of the strengths of Random Forest is its interpretability. We used feature importance scores to identify which variables most influenced the model’s decisions. Top features included Frequent Flyer status, Preferred Property Star, and Age, giving clear business insights into what matters most for offer acceptance.

The final model performed with over 90% accuracy and a strong F1 and ROC AUC score. We saved the trained model as a .pkl file for deployment or reuse. This project demonstrates a complete machine learning workflow — from raw data to model interpretation — and provides actionable insights for real-world business applications.
