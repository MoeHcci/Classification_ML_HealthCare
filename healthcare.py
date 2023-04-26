'''
Section 1.0
-----------
Context:
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, 
and smoking status. Each row in the data provides relavant information about the patient.

Attribute Information
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient


About the model: 
-The model does not contain data for ages < 18 and BMI > 60 f

'''
#########################################################################################################################################################

'''
Section 2.0
-----------
Import all required libraries 
'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

#########################################################################################################################################################
'''
Section 3.0
-----------
Import the data
'''
df = pd.read_csv("C:\\Users\\PZ4L6Q\\Documents\\Learnings\\Projects\\Project_1\\healthcare-dataset-stroke-data.csv")
# print(df.head())

#########################################################################################################################################################
'''
Section 4.0
-----------
Perform (Exploratory Data Analysis) EDA on the data. 
'''


'''
Section 4.1
-----------
Drop columns if required
'''
df = df.drop('id', axis =1 ) #-> this column will not be useful during the training of the ML algorithm 
# print(df.head())


'''
Section 4.2
-----------
Analyze Each column & View the unique columns of 'SELECTED' columns . 
    Then, go over each column and to replace all of the white spaces by NaN.
    Finally, make a conclusion in regards of each of the 'SELECTED analyzed the columns
    Selected columns ->  gender, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke
    Unselected colummns (requires individual analyzation ) -> age, avg_glucose_level

'''

# df['gender'] = df['gender'].replace(r'^\s*$', np.nan, regex=True)
# print(df['gender'].unique()) 

##Conclusion: -> Feature Column -> Content: ['Male' 'Female' 'Other'] -> Requires OHE

# df['age'] = df['age'].replace(r'^\s*$', np.nan, regex=True)
# print(df['age'].unique()) #-> Feature Column  -> There is a lot of ages must analyze sepreatly  
##Conclusion: -> Feature Column  -> There is a lot of ages must analyze sepreatly  

# df['hypertension'] = df['hypertension'].replace(r'^\s*$', np.nan, regex=True)
# print(df['hypertension'].unique()) #->  Feature Column -> Content: [0 1] -> No need for OHE
##Conclusion: -> Feature Column -> Content: [0 1] -> No need for OHE

# df['heart_disease'] = df['heart_disease'].replace(r'^\s*$', np.nan, regex=True)
# print(df['heart_disease'].unique()) 
##Conclusion: -> Feature Column -> Content: [1 0] -> No need for OHE 

# df['ever_married'] = df['ever_married'].replace(r'^\s*$', np.nan, regex=True)
# print(df['ever_married'].unique()) #->  Feature Column -> Content: ['Yes' 'No'] -> Requires OHE 
##Conclusion: -> Feature Column -> ['Yes' 'No']. Will require OHE

# df['work_type'] = df['work_type'].replace(r'^\s*$', np.nan, regex=True)
# print(df['work_type'].unique()) #->  Feature Column -> Content:['Private' 'Self-employed' 'Govt_job' 'children' 'Never_worked'] -> Requires OHE
##Conclusion: -> Feature Column -> ['Private' 'Self-employed' 'Govt_job' 'children' 'Never_worked']. Will require OHE

# df['Residence_type'] = df['Residence_type'].replace(r'^\s*$', np.nan, regex=True)
# print(df['Residence_type'].unique())
##Conclusion: -> Feature Column  -> ['Urban' 'Rural'] -> will require  OHE

# df['avg_glucose_level'] = df['avg_glucose_level'].replace(r'^\s*$', np.nan, regex=True)
# print(df['avg_glucose_level'].unique()) 
#Conclusion: -> Feature Column -> [228.69 202.21 105.92 ...  82.99 166.29  85.28]. There is a lot of avg_glucose_levels must analyze sepreatl

# df['bmi'] = df['bmi'].replace(r'^\s*$', np.nan, regex=True)
# print(df['bmi'].unique())   
##Conclusion: -> Feature Column -> There is a lot of avg_glucose_levels must analyze sepreatl

# df['smoking_status'] = df['smoking_status'].replace(r'^\s*$', np.nan, regex=True)
# print(df['smoking_status'].unique()) 
##Conclusion: -> Feature Column -> ['formerly smoked' 'never smoked' 'smokes' 'Unknown']. Will require OHE

# df['stroke'] = df['stroke'].replace(r'^\s*$', np.nan, regex=True)
# print(df['stroke'].unique())  
##Conclusion: -> Label Column -> This is the lablel data [1 0]. It is categorical. Therefore we will apply a classification problem



'''
Section 4.3
-----------
Analze the rows
-Drop empty rows if no additonal fillings can be made
-I believe only column bmi has empty rows
-Dropping rows that will not impact the over all quality of the ML model
'''
# print(df['bmi'].isna().sum()) #bmi has 201 Nan. Therefore out of 5110 dropping them will not impact the data significantly 
# print(len(df)) # -> 5110
df = df.dropna() 
# print(len(df)) # -> 4909


###Drop Rows From Columns, that do not provide significant infomration
df = df[df.gender != 'Other'] #-> Drop because there is only 1 input as 'Other'
df = df[df.work_type != 'Never_worked'] #-> Drop because there is a limited #inofmration
df = df[df.work_type != 'children'] #-> The ML will mainly focus on Adults not on childern
df = df[df.age >= 18] #-> The ML will mainly focus on Adults +18 not on childern
df = df[df.bmi < 60] #-> The ML will mainly focus on Adults +18 & have a bmi <60

###The dataset to work with is 
print(df.head())


'''
Section 4.4
-----------
Create Mathematical Conclusions 
'''
###0. General varilables to identify 
df_number_rows = df.shape[0] #The number of rows in the dataset 

###1. View the balance of the data & the stroke column
label_value_counts_stroke = df['stroke'].value_counts()
# print(100*(label_value_counts_stroke/df_number_rows) ) 
# print(label_value_counts_stroke) 
###Conclusion -> There are 95% of people do not have a stroke & 5% have a stroke. The data is cleary inbalanced  

###2. View the balance of the data for gender
label_value_counts_gender= df['gender'].value_counts()
# print(100*(label_value_counts_gender/df_number_rows) ) 
# print(label_value_counts_gender) 
###Conclusion -> The data contains 2475 females (61.035758%) & 1580 males (38.946242%)

###3. View the information for the age column
label_value_counts_age= df['age'].describe()
# print(label_value_counts_age)
###Conclusion -> The average Age is 49.9 years, with a SDS of 17.7 years, youngest is 18 (due to constraint on the data), and oldest is 82

###4. View the balance of the datafor hypertension
label_value_counts_hypertension = df['hypertension'].value_counts()
# print(100*(label_value_counts_hypertension/df_number_rows) ) 
# print(label_value_counts_hypertension) 
###Conclusion -> The data contains 3609(89.001233%) without hypertension & 446(10.998767%) with hypertension

###5. View the balance of the datafor heart_disease
label_value_counts_heart_disease = df['heart_disease'].value_counts()
# print(100*(label_value_counts_heart_disease/df_number_rows) ) 
# print(label_value_counts_heart_disease) 
###Conclusion -> The data contains 3813(94.032%) without heart_disease & 242(5.69%) with heart_disease

###6. View the balance of the datafor ever_married
label_value_counts_ever_married = df['ever_married'].value_counts()
# print(100*(label_value_counts_ever_married/df_number_rows) ) 
# print(label_value_counts_ever_married) 
###Conclusion -> The data contains 3193(78.74%) married people & 862(21.25%) not married

###7. View the balance of the datafor work_type
label_value_counts_work_type = df['work_type'].value_counts()
# print(100*(label_value_counts_work_type/df_number_rows) ) 
# print(label_value_counts_work_type) 
###Conclusion -> The data contains 2669 (65%) 'private' work_type people,  762 (18.79%) 'Self-employed' work_type people, &  624 (13.38%) 'Govt_job' work_type people

###8. View the balance of the datafor Residence_type
label_value_counts_Residence_type = df['Residence_type'].value_counts()
# print(100*(label_value_counts_Residence_type/df_number_rows) ) 
# print(label_value_counts_Residence_type) 
###Conclusion -> The data contains 2064 (50.900123%) live in Urban env & 1991(49.099877%) live in Rural env

###9. View the information for the avg_glucose_level column
label_value_counts_avg_glucose_level= df['avg_glucose_level'].describe()
# print(label_value_counts_avg_glucose_level)
###Conclusion -> The average  is 107, with a high SDS of 46, smallest is 55, and largest is 271

###10. View the information for the bmi column
label_value_counts_bmi= df['bmi'].describe()
# print(label_value_counts_bmi)
###Conclusion -> The average  is 30, with a decent SDS of 6.7, lowest is 11.3, and highest is 59.7

###11. View the balance of the datafor smoking_status
label_value_counts_smoking_status = df['smoking_status'].value_counts()
# print(100*(label_value_counts_smoking_status/df_number_rows) ) 
# print(label_value_counts_smoking_status) 
###Conclusion -> The data contains: 1706(42%) never smoked, 813 (20%) formerly smoked, 812(20%) unknown, 724(17.8%) smokes


###2. View the correlation between the 'Label' & all the 'Numerical' columns. 
correlation = df.corrwith(df['stroke']).sort_values(ascending=True)
# print(correlation[:-1]) #Make sure not to include the df['stroke'] it self, because it will have 100% correlation
#Conclusion
    #The correlation smallest to highest order of correlation is -> bmi, heart_disease, avg_glucose_level, age. This is only the numerical data



'''
Section 4.5
-----------
Create Plots and Add a Conclusion for Each Plot
'''
# ###1. View the balance of the 'Label' data 
# ##sns.countplot(data=df, x=df['stroke']) #This method does not show the numbers on top of the bars

# ax = sns.countplot(x='stroke', data=df)
# ax.bar_label(ax.containers[0])
# plt.show()

# ##Conclusion
#     ##The data is not balanced. 5.4% of the data is a 0. Therefore a ML algorithm of imbalanced data must be selected
#     ##When creating the ML model we either need to over sample or under sample  has to be done to obtain best results.

# ###2. Create a correlation plot between 'Label' & all the 'Numerical' columns. 
# df.corrwith(df['stroke']).sort_values(ascending=True)[:-1].plot(kind='bar')
# plt.show()
# ##Conclusion
#     ##The correlation smallest to highest order of correlation is -> bmi, heart_disease, avg_glucose_level, age. This is only the numerical data
#     ##bmi                  0.004450
#     ##heart_disease        0.130184
#     ##hypertension         0.132640
#     ##avg_glucose_level    0.135220
#     ##age                  0.235597

# ##3. Create a heatmap of the correlations between all features. 
# sns.heatmap(df.corr(), annot=True)
# plt.show()
# #Conclusion
#     #Notiable correlations is between: age & all the other numerical features


# ##4.1. View the balance of the 'gender' data 
# ax = sns.countplot(x='gender', data=df, hue=df['stroke'])
# ax.bar_label(ax.containers[0])
# ax.bar_label(ax.containers[1])
# plt.show()
# #Conclusion
#     #For males -> 89/(1491+89) have strokes = 5.6% 
#     #For females -> 119/(2356+119) have strokes = 4.8% 

# ##4.2. View the balance of the 'hypertension' data 
# ax = sns.countplot(x='hypertension', data=df, hue=df['stroke'])
# ax.bar_label(ax.containers[0])
# ax.bar_label(ax.containers[1])
# plt.show()
# #Conclusion
#     #For hypertension yes -> 60/(386+60)) have strokes = 13% 
#     #For hypertension no -> 148/(148+3461) have strokes = 4.1% 

# ##4.3. View the balance of the 'heart_disease' data 
# ax = sns.countplot(x='heart_disease', data=df, hue=df['stroke'])
# ax.bar_label(ax.containers[0])
# ax.bar_label(ax.containers[1])
# plt.show()
# #Conclusion
#     #For heart_disease yes -> 40/(40+202)) have strokes = 16% 
#     #For heart_disease no -> 168/(168+3645) have strokes = 4.4% 


# ##4.4. View the balance of the 'ever_married' data 
# ax = sns.countplot(x='ever_married', data=df, hue=df['stroke'])
# ax.bar_label(ax.containers[0])
# ax.bar_label(ax.containers[1])
# plt.show()
# #Conclusion
#     #For ever_married yes -> 186/(186+3007)) have strokes = 5% 
#     #For ever_married no -> 22/(22+840) have strokes = 2.5% 


# ##4.5. View the balance of the 'work_type' data 
# ax = sns.countplot(x='work_type', data=df, hue=df['stroke'])
# ax.bar_label(ax.containers[0])
# ax.bar_label(ax.containers[1])
# plt.show()
# #Conclusion
#     #For Gov_job  -> 28/(28+596)) have strokes = 4.4% 
#     #For Self-employed -> 53/(53+709) have strokes = 7% 
#     #For Private -> 127/(127+2542) have strokes = 4.7% 

# ##4.6. View the balance of the 'Residence_type' data 
# ax = sns.countplot(x='Residence_type', data=df, hue=df['stroke'])
# ax.bar_label(ax.containers[0])
# ax.bar_label(ax.containers[1])
# plt.show()
# #Conclusion
#     #For Urban  -> 109/(109+1955)) have strokes = 5.3% 
#     #For Rural-> 99/(99+1892) have strokes = 5% 

# ##4.7. View the balance of the 'smoking_status' data 
# ax = sns.countplot(x='smoking_status', data=df, hue=df['stroke'])
# ax.bar_label(ax.containers[0])
# ax.bar_label(ax.containers[1])
# plt.show()
# #Conclusion
#     #For Formerly_smoked  -> 57/(57+756)) have strokes = 7% 
#     #For never_smoked -> 84/(84+1622) have strokes = 4.9% 
#     #For smokes -> 39/(39+685) have strokes = 5.3% 
#     #For unknown -> 28/(28+784) have strokes = 3.4% 

###6. Create a bar plots between notible columns & the label as hue For Numerical Columns 

# ##6.1 Create a hisplot for the Age column 
# sns.histplot(data=df, x="age", binwidth=10, hue=df['stroke'], kde=True)
# plt.show()
# #Conclusion
# #    1. The older you get the higher changes of a stroke you have which makes sense 

# ##6.2 Create a hisplot for the Age column 
# sns.histplot(data=df, x="avg_glucose_level", binwidth=10, hue=df['stroke'],  kde=True)
# plt.show()
# ##Conclusion
#    #1. When the body has high avg glucose levels, means it is not producing inslin properly and more people with strokes of higher avg glucose levels 

# ##6.3 Create a hisplot for the bmi column 
# sns.histplot(data=df, x="bmi", binwidth=10, hue=df['stroke'],  kde=True)
# plt.show()
# ##Conclusion
# #  # 1. The higher the BMI the higher changes of a stroke 


# ##7. Scatter Plots 
# ###7.1 age & bmi with stroke
# sns.scatterplot(data=df,x=df['age'],y=df['bmi'], hue='stroke',   alpha=.4)
# plt.show()
# #Conclusion
# #   1. The higher the BMI & older age have higher chances of a stroke

# ###7.2 age & avg_glucose_level with stroke
# sns.scatterplot(data=df,x=df['age'],y=df['avg_glucose_level'], hue='stroke',   alpha=.4)
# plt.show()
# #Conclusion
# #   1. The higher the age & avg_glucose_level have  higher chances of a stroke


#########################################################################################################################################################
'''
Section 5.0
-----------
Prep The Data for ML Analysis  

The data we are working with is inbalanced. We need to figure out what to do with it: 
1. Random under-sampling --> Removing data, could impact the accuracy of the model, because valuable information will be removed
2. Random over-sampling --> Adding more data, good choice when we do not have a ton of data, Do not create exact copies, It increases the likelihood of overfitting since it replicates the minority class events. 
3. NearMiss

Notes: 
-The data is highly inbalanced. Therefore, a metric like 'Accuracy Score' can be misleading to use. We need to aovid that to avoid The Metrix Trap
    -The 'Accuracy' classifier will always “predicts” the most common class without performing any analysis of the features, and it will have a high accuracy rate, obviously not the correct one.
    -Accuracy = TP + TN / (All the data) -> Majority of data will be TP in an inbalanced data. Therefore, Accuracy will always be high

    
-Based on the available online infomration we have the following options: 

-Utilize the SMOTE Algorithm & Apply 'Over-Sampling'
    -SMOTE (Synthetic Minority Oversampling Technique) works by randomly picking a point from the minority class and 
        computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.

-Do not 'Over-Sample' and use SVC with class_weight = 'balanced

-Do not 'Over-Sample' & apply a decision tree

-Based on the available online infomration 'accuracy' seems to be a poor choice of a metrix. Therefore, we will construct a 
    Confusion Matrix, 
    Precision, --> TP / (TP+FP)
    Recall, TP / Tp + FN
    F1-Score, (2*Precision*Recall) / (Precision+Recall)
    ROC and AUC

    
-Based on the available options we selected the following: 
-Utilize the SMOTE Algorithm & Apply 'Over-Sampling'. Also, we can use a Random-Oversampling (Copy-Pasting your data set) but we will go with SMOTE
    -SMOTE (Synthetic Minority Oversampling Technique) works by randomly picking a point from the minority class and 
        computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.
    -SMOTE plots points between the under represented class by using K-NN, which creates addtional points between the area of the miniority class
    -SMOTE should be applied only to the test data. Therefore, a split must happen first  
        -Focus on SMOTEs paramters of weights, scoring, smote_ratio

-Utilize multiple Classification algorithms & create findings based on the selected metrices of evaluations
    Confusion Matrix, 
    Precision, --> TP / (TP+FP)
    Recall, TP / Tp + FN
    F1-Score, (2*Precision*Recall) / (Precision+Recall)
    ROC and AUC

'''


'''
Section 5.1
-----------
Categorize the data & Perfrom OHE (Only on Categorical Data)
'''

##Add the 'Label' to a new DataFrame 
y_label_main =  df[['stroke']].copy()

##Create a Numerical DataFrame (Won't experince OHE) & A Categorical DataFrame (Will experince OHE)
X_Numerical = df[['age', 'hypertension','heart_disease','avg_glucose_level', 'bmi']].copy()
X_Categorical = df[['gender', 'ever_married','work_type','Residence_type', 'smoking_status']].copy()

###Perfrom the OHE on the Categorical DataFrame
from sklearn.preprocessing import OneHotEncoder
##Creating an instance of OneHotEncoder()
encoder_object = OneHotEncoder() 
##Fit to data, then transform it.
Fitted_Data = encoder_object.fit_transform(X_Categorical)
##Add Column Names
column_name = encoder_object.get_feature_names_out(X_Categorical.columns)
##DataFrame from a scipy sparse matrix.
    ###scipt sparse matrix is a matrix only filled in the locations it is suppose too
    ##A sparse matrix is a matrix in which most of the elements have zero value and thus efficient ways of storing such matrices are required.
X_Categorical = pd.DataFrame.sparse.from_spmatrix(Fitted_Data, columns=column_name)


##Adding a column to merge on for both DataFrames. Otherwise you will be left with Nan if the regular concat method is used
X_Categorical['index'] = range(1, len(X_Categorical) + 1)
X_Numerical['index'] = range(1, len(X_Numerical) + 1)


##Merging the DataFrames
X_features_main = X_Categorical.merge(X_Numerical, on = 'index', how = 'left')

###Dropping the index Column 
X_features_main = X_features_main.drop(['index'], axis=1)



'''
Section 5.2
-----------
Split the data
The data will split to 70% Training | 30 % Validation (holdout)
    -Training will be done on the training data 
    -Initial evaluation / hyperparamters tuning will be done on the validation data 
    -Final model evaluation, which has no further evaulation after that will be done on the holdout data

'''
###Split the data into Training / Holdset 
from sklearn.model_selection import train_test_split
X_train, X_test_validation, y_train, y_test_validation = train_test_split(X_features_main, y_label_main, test_size=0.3, random_state=101) #tuple and packing



'''
Splitting Note: 
-The data is split into a training and a testing set ONLY. Commonly we split the data into a training set, a testing set, and a validation set. However, this approach  was followed in this project, because of the following: 
    -Cross validation is used to for the training and the testing of the data. Then, a final validation set (called test_validation set) is used for validating the score of the model 
    -If cross validation was not used &: 
        -The model's hyperparamters were tuned using the testing set. Then, a seperate validation  set will be required to evaluate the true score of the model. However when using GridSearchCV we the validation is not invovled  
'''

'''
Section 5.3
-----------
Apply a version SMOTE to create balance between classes, types to of SMOTE to consider: 

-SMOTE -> The synthetic data are created randomly between the two data 
-Borderline-SMOTE -> Makes synthetic data along the decision boundary between the two classes
-SVM-SMOTE -> the borderline area is approximated by the support vectors after training SVMs classifier on the original training set
    -SVM algorithm is used instead of a KNN to identify misclassified examples on the decision boundary
-Adaptive Synthetic Sampling (ADASYN) -> generates more synthetic points in regions where the density of minority class is low. 
    -> Also ADASYN generates few or none synthetic points, where the density of the minority class is high

Selected version of SMOTE -> The original SMOTE algoirthm     

There are multiple balancing options available and we need to select one or select both and compare them. Options are: 

Balancing Options 1: 
1. Oversample the minority class to make it equal to the Majority class 

Balancing Options 2: 
-A common suggestion is to combine SMOTE of the minority class with undersampling of the majoirty class, which is what the original paper on SMOTE suggested
    -This is to increase the effectiveness of handling the imbalanced class.
1. Oversample the minorty class, but instead of balancing the sata ensure that the minority class is around 10% of the majoirty class (e.g., 1000 for minority class)
    -we should attempty to see different pssibilities k_neighbors  
2. Perfrom undersamping in the majoirty class until it eqauls to 50% more of whatever the minority class was oversampled too (e.g., 2000 for majoirty class)
'''
from imblearn.over_sampling import SMOTE
###The count of the 'stroke' class prior to SMOTE -> 0 = 2697 & 1 = 141
# print(y_train['stroke'].value_counts())

###General SMOTE method with Balancing Option 1
sm_balanc1 = SMOTE(random_state=101) 
##random_state=42 -> To always have the same exact set of data that were transformed
X_train_sm_balanc1, y_train_sm_balanc1 = sm_balanc1.fit_resample(X_train, y_train)


###The count of the 'stroke' class prior to SMOTE -> 0 = 2697 & 1 = 2697
# print(y_sm_balanc1['stroke'].value_counts())

'''
Section 5.4
-----------
Scale the data
'''
import warnings
warnings.filterwarnings('ignore')
print(X_train_sm_balanc1.columns)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_sm_balanc1) 
    ###Scaler must be fitted to find the mean & standard deviation 
    ###This will use the X_train to train the model to scale all of its data to a similar range 
    ###We must use the training set not the test set to avoid dataleakage 
X_train_sm_balanc1 = scaler.transform(X_train_sm_balanc1)  #Scale the training set using the "scale" varilable. Overwrite the original X-train set
X_test_validation = scaler.transform(X_test_validation) #Scale the test_validation set using the "scale" varilable. 



#########################################################################################################################################################
'''
Section 6.0
-----------
Build each ML model, evaulte the model aganist the test set / adjust the hyperparamters, evaluate the model aganist the holdout set and make a final conclusion about the model

ML Models built are: 
-Logistic Regression
-KNN 
-SVM (SVC)
-Decision Trees (With Boosting)
-Random Forest
'''

'''
Section 6.1
-----------
The Logistic Regression Model 
'''

'''
Section 6.1.1
-----------
Conduct a Grid Search. Then, commnet out post GridSearch
'''

##The commented GridSearches. I have looked at recall, precision, and accuracy
##Accuracy was included, because the data was balanced used SMOTE
'''
.......................................................................................................................................................................
The Commented GridSearch  - > scoring='recall'
-----------
from sklearn.linear_model import LogisticRegression
ML_logistic_model = LogisticRegression()

from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
parameters = {'penalty' : ['l1','l2','elasticnet'],'C': np.logspace(-3,5,7),'l1_ratio': np.linspace(0,1,5),'solver'  : ['liblinear', 'saga','newton-cholesky'],'max_iter':[100,1000]} #Theses paramters we tune

ML_logistic_model_grid_model = GridSearchCV(ML_logistic_model, param_grid = parameters, scoring='recall', cv=10)                     
ML_logistic_model_grid_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

print("Tuned Hyperparameters :", ML_logistic_model_grid_model.best_params_) #-> {'C': 0.46415888336127775, 'l1_ratio': 0.0, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}
print("recall :",ML_logistic_model_grid_model.best_score_) #With GridsearchCV the is recall : 0.778265179677819
.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='precision'
-----------
from sklearn.linear_model import LogisticRegression
ML_logistic_model = LogisticRegression()

from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
parameters = {'penalty' : ['l1','l2','elasticnet'],'C': np.logspace(-3,5,7),'l1_ratio': np.linspace(0,1,5),'solver'  : ['liblinear', 'saga','newton-cholesky'],'max_iter':[100,1000]} #Theses paramters we tune

ML_logistic_model_grid_model = GridSearchCV(ML_logistic_model, param_grid = parameters, scoring='precision', cv=10)                     
ML_logistic_model_grid_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

print("Tuned Hyperparameters :", ML_logistic_model_grid_model.best_params_) #-> {'C': 0.46415888336127775, 'l1_ratio': 0.0, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}
print("precision :",ML_logistic_model_grid_model.best_score_) #With GridsearchCV the is precision : 0.7389412797408521
.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='accuracy'
-----------
from sklearn.linear_model import LogisticRegression
ML_logistic_model = LogisticRegression()

from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
parameters = {'penalty' : ['l1','l2','elasticnet'],'C': np.logspace(-3,5,7),'l1_ratio': np.linspace(0,1,5),'solver'  : ['liblinear', 'saga','newton-cholesky'],'max_iter':[100,1000]} #Theses paramters we tune

ML_logistic_model_grid_model = GridSearchCV(ML_logistic_model, param_grid = parameters, scoring='accuracy', cv=10)                     
ML_logistic_model_grid_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

print("Tuned Hyperparameters :", ML_logistic_model_grid_model.best_params_) #-> {'C': 0.46415888336127775, 'l1_ratio': 0.5, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}
print("accuracy :",ML_logistic_model_grid_model.best_score_) #With GridsearchCV the is accuracy : 0.7510231567374424
.......................................................................................................................................................................
'''

'''
Section 6.1.2
-----------
Based on the Grid Search build a Logistic Regression Model & Evaluate it For Recall, Precision, and Accuracy. Evaluated aganist the training set. 
The evaluation is done using a dataset that the model was never train on
'''

'''
Section 6.1.2.1 
-----------
Based on the Grid Search build a Logistic Regression Model & Evaluate it For Recall. 
'''
# from sklearn.linear_model import LogisticRegression
# ML_logistic_model = LogisticRegression(penalty ='l1',C=0.46415888336127775 , solver='liblinear', max_iter=100,l1_ratio=0.0) #Results from GridSearchCV
# ML_logistic_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_logistic = ML_logistic_model.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_logistic, labels=ML_logistic_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ML_logistic_model.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_logistic)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_logistic, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_logistic, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_logistic, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 20 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 827 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 47 people to have a stroke & they had a stroke. 
#     ##The model predicted 323 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =98%, recall =  72%, f1-score = 83% 
#     ##For "1"  having a stroke ---> precision =13%, recall =  70%, f1-score = 22% 
#     ##sample-weighted F1 score is  0.7944869493760204
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is  0.7376188755498763
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.8693929958698908

'''
Section 6.1.2.2 
-----------
Based on the Grid Search build a Logistic Regression Model & Evaluate it For Precision
'''
# from sklearn.linear_model import LogisticRegression
# ML_logistic_model = LogisticRegression(penalty ='l2',C=0.46415888336127775 , solver='liblinear', max_iter=100,l1_ratio=0.0) #Results from GridSearchCV
# ML_logistic_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_logistic = ML_logistic_model.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_logistic, labels=ML_logistic_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ML_logistic_model.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_logistic)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_logistic, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_logistic, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_logistic, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 20 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 827 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 47 people to have a stroke & they had a stroke. 
#     ##The model predicted 323 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =98%, recall =  72%, f1-score = 83% 
#     ##For "1"  having a stroke ---> precision =13%, recall =  70%, f1-score = 22% 
#     ##sample-weighted F1 score is  0.7944869493760204
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is  0.7376188755498763
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.8693929958698908
 

'''
Section 6.1.2.3 
-----------
Based on the Grid Search build a Logistic Regression Model & Evaluate it For Accuracy

'''
# from sklearn.linear_model import LogisticRegression
# ML_logistic_model = LogisticRegression(penalty ='l1',C=0.46415888336127775 , solver='liblinear', max_iter=100,l1_ratio=0.5) #Results from GridSearchCV
# ML_logistic_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_logistic = ML_logistic_model.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_logistic, labels=ML_logistic_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ML_logistic_model.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_logistic)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_logistic, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_logistic, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_logistic, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 20 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 827 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 47 people to have a stroke & they had a stroke. 
#     ##The model predicted 323 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =98%, recall =  72%, f1-score = 83% 
#     ##For "1"  having a stroke ---> precision =13%, recall =  70%, f1-score = 22% 
#     ##sample-weighted F1 score is  0.7944869493760204
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is  0.7376188755498763
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.8693929958698908



'''
Section 6.1.3
-----------
Logistic Regression Classification Model Conclusions 
-When recall, precision and accruacy were the scoring metricies the results were very similar 
'''






'''
Section 6.2
-----------
KNN  
'''

'''
Section 6.2.1
-----------
Conduct a Grid Search. Then, commnet out post GridSearch
'''



'''
.......................................................................................................................................................................
The Commented GridSearch  - > scoring='recall'
-----------

import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
from sklearn.neighbors import KNeighborsClassifier
ML_KNN_model = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
parameters = {'n_neighbors' : np.arange(1,30,4),'weights': ['uniform', 'distance'],'algorithm':['ball_tree', 'kd_tree', 'brute'],'p':[1, 2] } #Theses paramters we tune
ML_KNN_model_grid_model = GridSearchCV(ML_KNN_model, param_grid = parameters, scoring='recall', cv=10)                     
ML_KNN_model_grid_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data
print("Tuned Hyperparameters :", ML_KNN_model_grid_model.best_params_) #-> {'algorithm': 'ball_tree', 'n_neighbors': 1, 'p': 2, 'weights': 'uniform'}
print("recall :",ML_KNN_model_grid_model.best_score_) #With GridsearchCV the is recall : 0.9892551287346827



.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='precision'
-----------
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
from sklearn.neighbors import KNeighborsClassifier
ML_KNN_model = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
parameters = {'n_neighbors' : np.arange(1,30,4),'weights': ['uniform', 'distance'],'algorithm':['ball_tree', 'kd_tree', 'brute'],'p':[1, 2] } #Theses paramters we tune
ML_KNN_model_grid_model = GridSearchCV(ML_KNN_model, param_grid = parameters, scoring='precision', cv=10)                     
ML_KNN_model_grid_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data
print("Tuned Hyperparameters :", ML_KNN_model_grid_model.best_params_) #-> {'algorithm': 'ball_tree', 'n_neighbors': 1, 'p': 1, 'weights': 'uniform'}
print("precision :",ML_KNN_model_grid_model.best_score_) #With GridsearchCV the is precision : 0.9442581936754533



.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='accuracy'
-----------
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
from sklearn.neighbors import KNeighborsClassifier
ML_KNN_model = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
parameters = {'n_neighbors' : np.arange(1,30,4),'weights': ['uniform', 'distance'],'algorithm':['ball_tree', 'kd_tree', 'brute'],'p':[1, 2] } #Theses paramters we tune
ML_KNN_model_grid_model = GridSearchCV(ML_KNN_model, param_grid = parameters, scoring='accuracy', cv=10)                     
ML_KNN_model_grid_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data
print("Tuned Hyperparameters :", ML_KNN_model_grid_model.best_params_) #-> {'algorithm': 'ball_tree', 'n_neighbors': 1, 'p': 1, 'weights': 'uniform'}
print("accuracy :",ML_KNN_model_grid_model.best_score_) #With GridsearchCV the is accuracy : 0.963296571153714

.......................................................................................................................................................................
'''
'''
Section 6.2.2
-----------
Based on the Grid Search build a KNN Regression Model & Evaluate it For Recall, Precision, and Accuracy. Evaluated aganist the training set
'''


'''
Section 6.2.2.1 
-----------
Based on the Grid Search build a KNN Regression Model & Evaluate it For Recall
'''
# from sklearn.neighbors import KNeighborsClassifier
# ML_KNN_model = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree',p=2,weights='uniform') #Results from GridSearchCV
# ML_KNN_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_KNN = ML_KNN_model.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_KNN, labels=ML_KNN_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ML_KNN_model.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_KNN_model_classification_report  = classification_report(y_test_validation,y_predicted_KNN)
# print(ML_KNN_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_KNN, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_KNN, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_KNN, average='weighted', beta=0.5))


# ###Conclusion 
#     ##The model predicted 60  people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1072 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 7 people to have a stroke & they had a stroke. 
#     ##The model predicted 78 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  93%, f1-score = 94% 
#     ##For "1"  having a stroke ---> precision =8%, recall =  10%, f1-score = 9% 
#     ##sample-weighted F1 score is  0.8928732894083967
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is  0.8890792306991337
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is   0.8967585781481796
'''
Section 6.2.2.2 
-----------
Based on the Grid Search build a KNN Regression Model & Evaluate it For Precision
'''


# from sklearn.neighbors import KNeighborsClassifier
# ML_KNN_model = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree',p=1,weights='uniform') #Results from GridSearchCV
# ML_KNN_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_KNN = ML_KNN_model.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_KNN, labels=ML_KNN_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ML_KNN_model.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_KNN_model_classification_report  = classification_report(y_test_validation,y_predicted_KNN)
# print(ML_KNN_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_KNN, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_KNN, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_KNN, average='weighted', beta=0.5))


# ###Conclusion 
#     ##The model predicted 60 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1086 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 7 people to have a stroke & they had a stroke. 
#     ##The model predicted 67 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =94%, recall =  94%, f1-score = 95% 
#     ##For "1"  having a stroke ---> precision =10%, recall =  10%, f1-score = 10% 
#     ##sample-weighted F1 score is 0.8994980176762473
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is 0.8986634415009138
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is 0.9003379268608025


'''
Section 6.2.2.3 
-----------
Based on the Grid Search build a KNN Regression Model & Evaluate it For Accuracy
'''

# from sklearn.neighbors import KNeighborsClassifier
# ML_KNN_model = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree',p=1,weights='uniform') #Results from GridSearchCV
# ML_KNN_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_KNN = ML_KNN_model.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_KNN, labels=ML_KNN_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ML_KNN_model.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_KNN_model_classification_report  = classification_report(y_test_validation,y_predicted_KNN)
# print(ML_KNN_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_KNN, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_KNN, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_KNN, average='weighted', beta=0.5))




# ###Conclusion 
#     ##The model predicted 60 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1086 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 7 people to have a stroke & they had a stroke. 
#     ##The model predicted 64 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  94%, f1-score = 95% 
#     ##For "1"  having a stroke ---> precision =10%, recall =  10%, f1-score = 10% 
#     ##sample-weighted F1 score is 0.8994980176762473
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is  0.8986634415009138
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.9003379268608025

'''
Section 6.2.3
-----------
KNN Classification Model Conclusions 
-When recall, precision and accruacy were the scoring metricies the results were very similar 
'''

'''
Section 6.3
-----------
SVM (SVC)
'''
# Conduct the analysis for the SVM (SVC) here

'''
Section 6.3.1
-----------
Conduct a Grid Search. Then, commnet out post GridSearch
'''



'''
.......................................................................................................................................................................
The Commented GridSearch  - > scoring='recall'
-----------
from sklearn.svm import SVC
ML_SVM_model = SVC()


###Import the GridSearchCV
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
parameters = {'kernel' : ['rbf', 'poly','linear'],'C': [0.001,0.01, 0.1, 1, 10, 100, 100],'gamma':['scale', 'auto']  ,'degree' : [1,2,3,4]} #Theses paramters we tune

ML_SVM_model_GridSearchCV = GridSearchCV(ML_SVM_model, param_grid = parameters, scoring='recall', cv=10, n_jobs=-1)   #n_jobs=-1 -1 means using all processors                  
ML_SVM_model_GridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1) #fitting the model to the X and y trained data

print("Tuned Hyperparameters :", ML_SVM_model_GridSearchCV.best_params_) #-> Tuned Hyperparameters : {'C': 0.01, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
print("recall :",ML_SVM_model_GridSearchCV.best_score_) #With GridsearchCV the recall 0.9944430676029189


.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='precision'
-----------
from sklearn.svm import SVC
ML_SVM_model = SVC()


###Import the GridSearchCV
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
parameters = {'kernel' : ['rbf', 'poly','linear'],'C': [0.001,0.01, 0.1, 1, 10, 100, 100],'gamma':['scale', 'auto']  ,'degree' : [1,2,3,4]} #Theses paramters we tune

ML_SVM_model_GridSearchCV = GridSearchCV(ML_SVM_model, param_grid = parameters, scoring='precision', cv=10, n_jobs=-1)   #n_jobs=-1 -1 means using all processors                  
ML_SVM_model_GridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1) #fitting the model to the X and y trained data

print("Tuned Hyperparameters :", ML_SVM_model_GridSearchCV.best_params_) #-> Tuned Hyperparameters : {'C': 100, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
print("precision :",ML_SVM_model_GridSearchCV.best_score_) #With GridsearchCV the precision 0.9956944398919646



.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='accuracy'
-----------
from sklearn.svm import SVC
ML_SVM_model = SVC()


###Import the GridSearchCV
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
parameters = {'kernel' : ['rbf', 'poly','linear'],'C': [0.001,0.01, 0.1, 1, 10, 100, 100],'gamma':['scale', 'auto']  ,'degree' : [1,2,3,4]} #Theses paramters we tune

ML_SVM_model_GridSearchCV = GridSearchCV(ML_SVM_model, param_grid = parameters, scoring='accuracy', cv=10, n_jobs=-1)   #n_jobs=-1 -1 means using all processors                  
ML_SVM_model_GridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1) #fitting the model to the X and y trained data

print("Tuned Hyperparameters :", ML_SVM_model_GridSearchCV.best_params_) #-> Tuned Hyperparameters : {'C': 100, 'degree': 1, 'gamma': 'scale', 'kernel': 'rbf'}
print("accuracy :",ML_SVM_model_GridSearchCV.best_score_) #With GridsearchCV the accuracy 0.946993746993747



.......................................................................................................................................................................
'''



'''
Section 6.3.2
-----------
Based on the Grid Search build a SVM-SVC Classification Model & Evaluate it For Recall, Precision, and Accuracy. Evaluated aganist the training set
'''

'''
Section 6.3.2.1 
-----------
Based on the Grid Search build a SVM-SVC Classification Model & Evaluate it For Recall
'''
# from sklearn.svm import SVC
# ML_SVM_model = SVC(C=0.01, degree=3,gamma='auto',kernel='poly' )#Results from GridSearchCV
# ML_SVM_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_SVM_SVC = ML_SVM_model.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_SVM_SVC, labels=ML_SVM_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ML_SVM_model.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_SVM_SVC)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_SVM_SVC, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_SVM_SVC, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_SVM_SVC, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 6 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 353 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 61 people to have a stroke & they had a stroke. 
#     ##The model predicted 797 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =98%, recall =  31%, f1-score = 47% 
#     ##For "1"  having a stroke ---> precision =7%, recall =  91%, f1-score = 13% 
#     ##sample-weighted F1 score is  0.44936334659582683
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is   0.35123633883928923
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.6497450276844271


'''
Section 6.3.2.2 
-----------
Based on the Grid Search build a SVM-SVC Classification Model & Evaluate it For Precision
'''

# from sklearn.svm import SVC
# ML_SVM_model = SVC(C=100, degree=2,gamma='scale',kernel='poly' )#Results from GridSearchCV
# ML_SVM_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_SVM_SVC = ML_SVM_model.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_SVM_SVC, labels=ML_SVM_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ML_SVM_model.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_SVM_SVC)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_SVM_SVC, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_SVM_SVC, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_SVM_SVC, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 67 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1147 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 0 people to have a stroke & they had a stroke. 
#     ##The model predicted 3 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =94%, recall =  100%, f1-score = 97% 
#     ##For "1"  having a stroke ---> precision =0.00%, recall =  0.00%, f1-score = 0.00% 
#     ##sample-weighted F1 score is  0.9169659379879235
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is   0.9321067584294165
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.9023091397783262

'''
Section 6.3.2.3 
-----------
Based on the Grid Search build a SVM-SVC Classification Model & Evaluate it For Accuracy
'''

# from sklearn.svm import SVC
# ML_SVM_model = SVC(C=100, degree=1,gamma='scale',kernel='rbf' )#Results from GridSearchCV
# ML_SVM_model.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_SVM_SVC = ML_SVM_model.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_SVM_SVC, labels=ML_SVM_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ML_SVM_model.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_SVM_SVC)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_SVM_SVC, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_SVM_SVC, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_SVM_SVC, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 58 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1089 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 9 people to have a stroke & they had a stroke. 
#     ##The model predicted 61 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  95%, f1-score = 95% 
#     ##For "1"  having a stroke ---> precision =0.13%, recall =  0.13%, f1-score = 0.13% 
#     ##sample-weighted F1 score is  0.9032253160652699
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is   0.9026200401393151
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.903834190135805


'''
Section 6.3.3
-----------
SVM-SVC  Classification Model Conclusions 
-When recall is the metric of choice 
    -The number of predicting people not to have a stroke and they end up having a stroke is lowest at 6 and that is the lowest between all three metrices 
    -The model predicted 797 to have a stroke and they did not have a stroke and that is the highest between all three metrices. 
    -This is impacting the over all metrices:
    -Sample-weighted F1 score is  0.44936334659582683
    -Sample-weighted Beta score With a F(2), emphasis on Recall is   0.35123633883928923
    -Sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.6497450276844271


-When precision is the metric of choice 
    -The number of predicting people not to have a stroke and they end up having a stroke is lowest at 67 and that is the highest between all three metrices 
    -The model predicted 3 to have a stroke and they did not have a stroke and that is the lowest between all three metrices 
    

-When accuracy is the metric of choice 
    -The number of predicting people not to have a stroke and they end up having a stroke is lowest at 58 and that is the middle between all three metrices 
    -The model predicted 61 to have a stroke and they did not have a stroke and that is the middle between all three metrices 
'''




'''
Section 6.4
-----------
Decision Trees (With Ada Boost)
'''
# Conduct the analysis for the Decision Trees (With Ada Boost)

'''
Section 6.4.1
-----------
Conduct a Grid Search. Then, commnet out post GridSearch
'''



'''
.......................................................................................................................................................................
The Commented GridSearch  - > scoring='recall'
-----------
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
ml_model = AdaBoostClassifier()
parameters = { 'n_estimators' : [10,50,100], 'learning_rate':[0.01, 0.1, 1]} #Theses paramters we tun for the AdaBoostClassifier
ML_AdaBoostClassifier_model_GridSearchCV = GridSearchCV(ml_model, param_grid = parameters, scoring='recall', cv=10)                     
ML_AdaBoostClassifier_model_GridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1)
print("Tuned Hyperparameters :", ML_AdaBoostClassifier_model_GridSearchCV.best_params_) #-> Tuned Hyperparameters : {'learning_rate': 0.01, 'n_estimators': 10}
print("Recall :",ML_AdaBoostClassifier_model_GridSearchCV.best_score_) #With GridsearchCV the Recall : 0.9480903208040754

.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='precision'
-----------

###Import the GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
ml_model = AdaBoostClassifier()
parameters = { 'n_estimators' : [10, 50,100,500], 'learning_rate':[0.01, 0.1, 1,1.5,2]} #Theses paramters we tun for the AdaBoostClassifier
ML_AdaBoostClassifier_model_GridSearchCV = GridSearchCV(ml_model, param_grid = parameters, scoring='precision', cv=10)                     
ML_AdaBoostClassifier_model_GridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1)
print("Tuned Hyperparameters :", ML_AdaBoostClassifier_model_GridSearchCV.best_params_) #-> Tuned Hyperparameters : {'learning_rate': 0.1, 'n_estimators': 1000}
print("precision :",ML_AdaBoostClassifier_model_GridSearchCV.best_score_) #With GridsearchCV the precision : 0.9981314490407038

.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='accuracy'
-----------

###Import the GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
ml_model = AdaBoostClassifier()
parameters = { 'n_estimators' : [10, 50,100,500], 'learning_rate':[0.01, 0.1, 1,1.5,2]} #Theses paramters we tun for the AdaBoostClassifier
ML_AdaBoostClassifier_model_GridSearchCV = GridSearchCV(ml_model, param_grid = parameters, scoring='accuracy', cv=10)                     
ML_AdaBoostClassifier_model_GridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1)
print("Tuned Hyperparameters :", ML_AdaBoostClassifier_model_GridSearchCV.best_params_) #-> Tuned Hyperparameters :  {'learning_rate': 1.5, 'n_estimators': 100}
print("accuracy :",ML_AdaBoostClassifier_model_GridSearchCV.best_score_) #With GridsearchCV the accuracy : 0.9642448292448293
.......................................................................................................................................................................
'''



'''
Section 6.4.2
-----------
Based on the Grid Search build a Decision Trees (With Ada Boost) Classification Model & Evaluate it For Recall, Precision, and Accuracy. Evaluated aganist the training set
'''
#The gridsearch was cooded here

'''
Section 6.4.2.1 
-----------
Based on the Decision Trees (With Ada Boost) Classification Model & Evaluate it For Recall
'''



# from sklearn.ensemble import AdaBoostClassifier
# ml_adaboost = AdaBoostClassifier(learning_rate=0.01, n_estimators=10)
# ml_adaboost.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_adaboost = ml_adaboost.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_adaboost, labels=ml_adaboost.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ml_adaboost.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_adaboost)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_adaboost, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_adaboost, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_adaboost, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 5 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 571 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 62 people to have a stroke & they had a stroke. 
#     ##The model predicted 579 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =99%, recall =  50%, f1-score = 66% 
#     ##For "1"  having a stroke ---> precision =10%, recall =  93%, f1-score = 18% 
#     ##sample-weighted F1 score is  0.6348617088426793
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is    0.5399927322091276
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.7875586685125681


'''
Section 6.4.2.2 
-----------
Based on the Decision Trees (With Ada Boost) Classification Model & Evaluate it For Precision
'''
# from sklearn.ensemble import AdaBoostClassifier
# ml_adaboost = AdaBoostClassifier(learning_rate=0.1, n_estimators=1000)
# ml_adaboost.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_adaboost = ml_adaboost.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_adaboost, labels=ml_adaboost.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ml_adaboost.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_adaboost)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_adaboost, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_adaboost, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_adaboost, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 66 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1149 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 1 people to have a stroke & they had a stroke. 
#     ##The model predicted 1 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  100%, f1-score = 97% 
#     ##For "1"  having a stroke ---> precision =50%, recall =  1%, f1-score = 3% 
#     ##sample-weighted F1 score is  0.9197721838961174
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is    0.9345909881447912
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is  0.9069511190834594

'''
Section 6.4.2.3 
-----------
Based on the Decision Trees (With Ada Boost) Classification Model & Evaluate it For Accuracy
'''
# from sklearn.ensemble import AdaBoostClassifier

# ml_adaboost = AdaBoostClassifier(learning_rate=1.5, n_estimators=100)
# ml_adaboost.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_adaboost = ml_adaboost.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_adaboost, labels=ml_adaboost.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ml_adaboost.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_adaboost)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_adaboost, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_adaboost, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_adaboost, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 63 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1141 people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 4 people to have a stroke & they had a stroke. 
#     ##The model predicted 9 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  99%, f1-score = 97% 
#     ##For "1"  having a stroke ---> precision =31%, recall =  6%, f1-score = 10% 
#     ##sample-weighted F1 score is   0.9215495713863847
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is   0.9327468373499279
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is   0.9128598304755745






'''
Section 6.5
-----------
Decision Trees (With Gradient Boost)
'''
# Conduct the analysis for the Decision Trees (With Gradient Boost)

'''
Section 6.5.1
-----------
Conduct a Grid Search. Then, commnet out post GridSearch
'''



'''
.......................................................................................................................................................................
The Commented GridSearch  - > scoring='recall'
-----------

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
ml_model = GradientBoostingClassifier()
parameters = { 'learning_rate':[0.01, 0.1,1],'n_estimators' : [10,40,100], 'max_depth': [2,4,8]  } #Theses paramters we tun for the AdaBoostClassifier
ML_GradientBoostingClassifier_model_GridSearchCV = GridSearchCV(ml_model, param_grid = parameters, scoring='recall', cv=10)                     
ML_GradientBoostingClassifier_model_GridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1)
print("Tuned Hyperparameters :", ML_GradientBoostingClassifier_model_GridSearchCV.best_params_) #-> Tuned Hyperparameters :{'learning_rate': 1, 'max_depth': 8, 'n_estimators': 40}
print("Recall :",ML_GradientBoostingClassifier_model_GridSearchCV.best_score_) #With GridsearchCV the Recall : 0.949997246316949
.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='precision'
-----------


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
ml_model = GradientBoostingClassifier()
parameters = { 'learning_rate':[0.01, 0.1,1],'n_estimators' : [10,40,100], 'max_depth': [2,4,8]  } #Theses paramters we tun for the AdaBoostClassifier
ML_GradientBoostingClassifier_model_GridSearchCV = GridSearchCV(ml_model, param_grid = parameters, scoring='precision', cv=10)                     
ML_GradientBoostingClassifier_model_GridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1)
print("Tuned Hyperparameters :", ML_GradientBoostingClassifier_model_GridSearchCV.best_params_) #-> Tuned Hyperparameters :{'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 100}
print("precision :",ML_GradientBoostingClassifier_model_GridSearchCV.best_score_) #With GridsearchCV the precision :  0.9958936926456656

.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='accuracy'
-----------

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
ml_model = GradientBoostingClassifier()
parameters = { 'learning_rate':[0.01, 0.1,1],'n_estimators' : [10,40,100], 'max_depth': [2,4,8]  } #Theses paramters we tun for the AdaBoostClassifier
ML_GradientBoostingClassifier_model_GridSearchCV = GridSearchCV(ml_model, param_grid = parameters, scoring='accuracy', cv=10)                     
ML_GradientBoostingClassifier_model_GridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1)
print("Tuned Hyperparameters :", ML_GradientBoostingClassifier_model_GridSearchCV.best_params_) #-> Tuned Hyperparameters : {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 100}
print("accuracy :",ML_GradientBoostingClassifier_model_GridSearchCV.best_score_) #With GridsearchCV the accuracy : 0.9683244004672575


.......................................................................................................................................................................
'''

'''
Section 6.5.2
-----------
Based on the Grid Search build a Decision Trees (With Gradient Boost) Classification Model & Evaluate it For Recall, Precision, and Accuracy. Evaluated aganist the training set
'''

'''
Section 6.5.2.1 
-----------
Based on the Decision Trees (With Gradient Boost) Classification Model & Evaluate it For Recall
'''

# from sklearn.ensemble import GradientBoostingClassifier

# ml_GradientBoosting = GradientBoostingClassifier(learning_rate=1,max_depth=8 ,n_estimators=40)
# ml_GradientBoosting.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_GradientBoosting = ml_GradientBoosting.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_GradientBoosting, labels=ml_GradientBoosting.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ml_GradientBoosting.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_GradientBoosting)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_GradientBoosting, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_GradientBoosting, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_GradientBoosting, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 63 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1134  people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 3 people to have a stroke & they had a stroke. 
#     ##The model predicted 16 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  98%, f1-score = 96% 
#     ##For "1"  having a stroke ---> precision =0%, recall =  0%, f1-score = 0% 
#     ##sample-weighted F1 score is   0.91396752216895
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is   0.9220425748047285
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is   0.9066139481987073





'''
Section 6.5.2.2 
-----------
Based on the Decision Trees (With Gradient Boost) Classification Model & Evaluate it For Precision
'''

# from sklearn.ensemble import GradientBoostingClassifier

# ml_GradientBoosting = GradientBoostingClassifier(learning_rate=0.1,max_depth=2 ,n_estimators=100)
# ml_GradientBoosting.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_GradientBoosting = ml_GradientBoosting.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_GradientBoosting, labels=ml_GradientBoosting.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ml_GradientBoosting.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_GradientBoosting)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_GradientBoosting, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_GradientBoosting, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_GradientBoosting, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 67 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1148  people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 0 people to have a stroke & they had a stroke. 
#     ##The model predicted 2 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  98%, f1-score = 96% 
#     ##For "1"  having a stroke ---> precision =0%, recall =  0%, f1-score = 0% 
#     ##sample-weighted F1 score is   0.9173773237139119
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is   0.9327589727357702
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is    0.9024947464989191


'''
Section 6.5.2.3 
-----------
Based on the Decision Trees (With Gradient Boost) Classification Model & Evaluate it For Accuracy
'''
# from sklearn.ensemble import GradientBoostingClassifier

# ml_GradientBoosting = GradientBoostingClassifier(learning_rate=0.1,max_depth=8 ,n_estimators=100)
# ml_GradientBoosting.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_GradientBoosting = ml_GradientBoosting.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_GradientBoosting, labels=ml_GradientBoosting.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ml_GradientBoosting.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_GradientBoosting)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_GradientBoosting, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_GradientBoosting, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_GradientBoosting, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 65 people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1132  people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 2 people to have a stroke & they had a stroke. 
#     ##The model predicted 18 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  99%, f1-score = 97% 
#     ##For "1"  having a stroke ---> precision =11%, recall =  3%, f1-score = 5.00% 
#     ##sample-weighted F1 score is    0.9145065235875544
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is   0.9251887499099176
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is    0.9047456195787078




'''
Section 6.6
-----------
Random Forest
'''
#Conduct the analysis for the Random Forest here

'''
Section 6.6.1
-----------
Conduct a Grid Search. Then, commnet out post GridSearch
'''



'''
.......................................................................................................................................................................
The Commented GridSearch  - > scoring='recall'
-----------
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
ml_model = RandomForestClassifier()
parameters = {'n_estimators': [50, 100, 200, 400], 'max_depth': [2, 5, 10, 20,  None] , 'max_features': ['sqrt', None], 'bootstrap':[True, False], 'n_jobs' : [-1] } #Theses paramters we tun for the AdaBoostClassifier
ML_RandomForestClassifier_GradiridSearchCV = GridSearchCV(ml_model, param_grid = parameters, scoring='accuracy', cv=10)                     
ML_RandomForestClassifier_GradiridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1)
print("Tuned Hyperparameters :", ML_RandomForestClassifier_GradiridSearchCV.best_params_) #-> Tuned Hyperparameters : {'bootstrap': True, 'max_depth': None, 'max_features': None, 'n_estimators': 200, 'n_jobs': -1}
print("accuracy :",ML_RandomForestClassifier_GradiridSearchCV.best_score_) #With GridsearchCV the Recall : 0.9518408371196475

.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='precision'
-----------
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
ml_model = RandomForestClassifier()
parameters = {'n_estimators': [50, 100, 200, 400], 'max_depth': [2, 5, 10, 20,  None] , 'max_features': ['sqrt', None], 'bootstrap':[True, False], 'n_jobs' : [-1] } #Theses paramters we tun for the AdaBoostClassifier
ML_RandomForestClassifier_GradiridSearchCV = GridSearchCV(ml_model, param_grid = parameters, scoring='accuracy', cv=10)                     
ML_RandomForestClassifier_GradiridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1)
print("Tuned Hyperparameters :", ML_RandomForestClassifier_GradiridSearchCV.best_params_) #-> Tuned Hyperparameters :{'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 100, 'n_jobs': -1}
print("accuracy :",ML_RandomForestClassifier_GradiridSearchCV.best_score_) #With GridsearchCV the precision : 0.9974018796051765




.......................................................................................................................................................................

.......................................................................................................................................................................
The Commented GridSearch  - > scoring='accuracy'
-----------
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')#  Will not show any of the warnings
ml_model = RandomForestClassifier()
parameters = {'n_estimators': [50, 100, 200, 400], 'max_depth': [2, 5, 10, 20,  None] , 'max_features': ['sqrt', None], 'bootstrap':[True, False], 'n_jobs' : [-1] } #Theses paramters we tun for the AdaBoostClassifier
ML_RandomForestClassifier_GradiridSearchCV = GridSearchCV(ml_model, param_grid = parameters, scoring='accuracy', cv=10)                     
ML_RandomForestClassifier_GradiridSearchCV.fit(X_train_sm_balanc1,y_train_sm_balanc1)
print("Tuned Hyperparameters :", ML_RandomForestClassifier_GradiridSearchCV.best_params_) #-> Tuned Hyperparameters :{'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 400, 'n_jobs': -1}
print("accuracy :",ML_RandomForestClassifier_GradiridSearchCV.best_score_) #With GridsearchCV the accuracy : 0.9731443688586546




.......................................................................................................................................................................
'''

'''
Section 6.6.2
-----------
Based on the Grid Search build a Random Forest Classification Model & Evaluate it For Recall, Precision, and Accuracy. Evaluated aganist the training set
'''


'''
Section 6.6.2.1 
-----------
Based on the Random Forest Classification Model Evaluate it For Recall
'''
# from sklearn.ensemble import RandomForestClassifier

# ml_RandomForestClassifier = RandomForestClassifier(bootstrap=True, max_depth=None, max_features=None, n_estimators=200, n_jobs=-1)
# ml_RandomForestClassifier.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_ml_RandomForestClassifier= ml_RandomForestClassifier.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_ml_RandomForestClassifier, labels=ml_RandomForestClassifier.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ml_RandomForestClassifier.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_ml_RandomForestClassifier)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_ml_RandomForestClassifier, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_ml_RandomForestClassifier, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_ml_RandomForestClassifier, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 63  people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1119  people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 4 people to have a stroke & they had a stroke. 
#     ##The model predicted 31 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  97%, f1-score = 96% 
#     ##For "1"  having a stroke ---> precision =11%, recall =  6%, f1-score = 8.00% 
#     ##sample-weighted F1 score is   0.911174890654461
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is    0.9180192518043426
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is   0.9047707132477102


'''
Section 6.6.2.2 
-----------
Based on the Random Forest Classification Model Evaluate it For Precision
'''
# from sklearn.ensemble import RandomForestClassifier

# ml_RandomForestClassifier = RandomForestClassifier(bootstrap=True, max_depth=10, max_features='sqrt', n_estimators=100, n_jobs=-1)
# ml_RandomForestClassifier.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_ml_RandomForestClassifier= ml_RandomForestClassifier.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_ml_RandomForestClassifier, labels=ml_RandomForestClassifier.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ml_RandomForestClassifier.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_ml_RandomForestClassifier)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_ml_RandomForestClassifier, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_ml_RandomForestClassifier, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_ml_RandomForestClassifier, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 63  people Not to have a stroke & they had a stroke. 
#     ##The model predicted 1140  people Not to have a stroke & they did not have a stroke. 
#     ##The model predicted 4 people to have a stroke & they had a stroke. 
#     ##The model predicted 10 to have a stroke and they did not have a stroke. 
#     ##For "0" Not having a stroke ---> precision =95%, recall =  99%, f1-score = 97% 
#     ##For "1"  having a stroke ---> precision =29%, recall =  6%, f1-score = 10.00% 
#     ##sample-weighted F1 score is  0.9210677287568355
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is    0.9320788143903541
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is   0.9123727003311718

'''
Section 6.6.2.3 
-----------
Based on the Random Forest Classification Model Evaluate it For Accuracy
'''
# from sklearn.ensemble import RandomForestClassifier


# ml_RandomForestClassifier = RandomForestClassifier(bootstrap=True, max_depth=None, max_features='sqrt', n_estimators=400, n_jobs=-1)
# ml_RandomForestClassifier.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

# y_predicted_ml_RandomForestClassifier= ml_RandomForestClassifier.predict(X_test_validation) #view the predicted results from the fitted model by predicting the X Test (X_validation)

# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix,classification_report
# cm = confusion_matrix(y_test_validation, y_predicted_ml_RandomForestClassifier, labels=ml_RandomForestClassifier.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ml_RandomForestClassifier.classes_)
# disp.plot()
# plt.show() 

# ##View the classification report 
# ML_logistic_model_classification_report  = classification_report(y_test_validation,y_predicted_ml_RandomForestClassifier)
# print(ML_logistic_model_classification_report)
# from sklearn.metrics import f1_score, fbeta_score
# print("sample-weighted F1 score is ", f1_score(y_test_validation, y_predicted_ml_RandomForestClassifier, average='weighted'))
# print ("sample-weighted Beta score With a F(2), emphasis on Recall is ",fbeta_score(y_test_validation, y_predicted_ml_RandomForestClassifier, average='weighted', beta=2.0))
# print ("sample-weighted Beta score With a F(0.5), emphasis on Precision is ",fbeta_score(y_test_validation, y_predicted_ml_RandomForestClassifier, average='weighted', beta=0.5))

# ###Conclusion 
#     ##The model predicted 66  people Not to have a stroke & they had a stroke.  False Negative
#     ##The model predicted 1142  people Not to have a stroke & they did not have a stroke.  True Postive
#     ##The model predicted 1 people to have a stroke & they had a stroke. True Negative 
#     ##The model predicted 8 to have a stroke and they did not have a stroke. False Positive
#     ##For "0" Not having a stroke ---> precision =95%, recall =  99%, f1-score = 97% 
#     ##For "1"  having a stroke ---> precision =11%, recall =  1%, f1-score = 3.00% 
#     ##sample-weighted F1 score is  0.916740551515319
#     ##sample-weighted Beta score With a F(2), emphasis on Recall is     0.9299959871069324
#     ##sample-weighted Beta score With a F(0.5), emphasis on Precision is    0.9046526074275244






'''
Section 8.0
-----------
Deploying the most optimum model "Decision Trees with Ada Boost" when the metric is recall
'''

from sklearn.ensemble import AdaBoostClassifier
ml_adaboost = AdaBoostClassifier(learning_rate=0.01, n_estimators=10)
ml_adaboost.fit(X_train_sm_balanc1,y_train_sm_balanc1 ) #fitting the model to the X and y trained data

from joblib import dump #allows you to dump (save) the model and load (load it back up)
dump(ml_adaboost, 'ModelDeployment_pki_file.pkl') #This model can be used for the API
dump(encoder_object, 'OneHotEncoder.pkl') #We want to save the 'encoder' because that is the object that we fitted and transformed our orginal data on
dump(scaler, 'scalar.pkl') #This model scaler converter used for the final model & can be sent to other people if they have Python



'''
Example of Using the Model based on the generated file 
Code -->

##1. Load the libraries
from joblib import load
import pandas as pd

##2. Use load to load both the model and the converter
loaded_model = load("ModelDeployment_pki_file.pkl") #The line others will use when they want to load your file 
loaded_scaler = load("scalar.pkl") #The line others will use when they want to load your file 


##3. Bringing the new data they need to test for: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status

###Input #1 "gender" ---> Enter Either 'Male' or 'Female' as a string
gender = 'Female'
###DO NOT CHANGE ANYTHING IN THE IF STATEMENTS
if gender == 'Male': 
    gender_male= 1
    gender_female = 0
elif gender == 'Female': 
    gender_male= 0
    gender_female = 1




###Input #2 "ever_married" ---> Enter Either 'Yes' or 'No' as a string
evermarried = 'Yes'
###DO NOT CHANGE ANYTHING IN THE IF STATEMENTS
if evermarried == 'No': 
    ever_married_No= 1
    ever_married_Yes = 0
elif evermarried == 'Yes': 
    ever_married_No= 0
    ever_married_Yes = 1



###Input #3 "work_type" ---> Enter Either 'Govt_job' or 'Private' or 'Self_employed' as a string
work_type = 'Private'
###DO NOT CHANGE ANYTHING IN THE IF STATEMENTS
if work_type == 'Govt_job': 
    work_type_Govt_job= 1
    work_type_Private = 0
    work_type_Self_employed = 0
elif work_type == 'Private': 
    work_type_Govt_job= 0
    work_type_Private = 1
    work_type_Self_employed = 0
elif work_type == 'Self_employed': 
    work_type_Govt_job= 0
    work_type_Private = 0
    work_type_Self_employed = 1


###Input #4 "Residence_type" ---> Enter Either 'Rural' or 'Urban' as a string
residence_type = 'Urban'
###DO NOT CHANGE ANYTHING IN THE IF STATEMENTS
if residence_type == 'Urban': 
    Residence_type_Rural= 0
    Residence_type_Urban = 1
elif residence_type == 'Rural': 
    Residence_type_Rural= 1
    Residence_type_Urban = 0




###Input #5 "smoking_status" ---> Enter Either 'unknown' or 'formerly_smoked' or 'never_smoked' or 'smokes' as a string
smoke_status = 'smokes'
###DO NOT CHANGE ANYTHING IN THE IF STATEMENTS
if smoke_status == 'unknown': 
    smoking_status_Unknown = 1
    smoking_status_formerly_smoked = 0
    smoking_status_never_smoked = 0
    smoking_status_smokes = 0
elif smoke_status == 'formerly_smoked': 
    smoking_status_Unknown = 0
    smoking_status_formerly_smoked = 1
    smoking_status_never_smoked = 0
    smoking_status_smokes = 0
elif smoke_status == 'never_smoked': 
    smoking_status_Unknown= 0
    smoking_status_formerly_smoked = 0
    smoking_status_never_smoked = 1
    smoking_status_smokes = 0
elif smoke_status == 'smokes': 
    smoking_status_Unknown= 0
    smoking_status_formerly_smoked = 0
    smoking_status_never_smoked = 0
    smoking_status_smokes = 1

###Input #6 "age" ---> Enter the age in years as an integer 
age = 70 

###Input #7 "hypertension" ---> Enter the hypertension as 0 or 1 (integer) . If there is hypertension then 1 and if not then 0
hypertension = 1 

###Input #8 "heart_disease" ---> Enter the heart_disease as 0 or 1 (integer). If there is heart_disease then 1 and if not then 0
heart_disease = 1 

###Input #9 "avg_glucose_level" ---> Enter the avg_glucose_level as an integer 
avg_glucose_level = 171.23 

###Input #10 "bmi" ---> Enter the bmi as an integer 
bmi = 40



input_continuous = [[gender_female,gender_male, ever_married_No, ever_married_Yes,work_type_Govt_job,work_type_Private,work_type_Self_employed,Residence_type_Rural,Residence_type_Urban , 
                     smoking_status_Unknown, smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes, age, hypertension,heart_disease,avg_glucose_level, bmi  ]]


###Side Note: The order of inputs is from the following: 
    # Index(['gender_Female', 'gender_Male', 'ever_married_No', 'ever_married_Yes',
    #    'work_type_Govt_job', 'work_type_Private', 'work_type_Self-employed',
    #    'Residence_type_Rural', 'Residence_type_Urban',
    #    'smoking_status_Unknown', 'smoking_status_formerly smoked',
    #    'smoking_status_never smoked', 'smoking_status_smokes', 'age',
    #    'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'],
    #   dtype='object')

##4. Scaling the inputted data
loaded_scaler.transform(input_continuous)
Transfromed_Scaled_data = loaded_scaler.transform(input_continuous)

##5. Based on the converted data predict the label 
y_predicted_from_loaded_model = loaded_model.predict(Transfromed_Scaled_data) #The model will predict y label based on the inputs 
print('Based on the inputs if the results is 0 means no Stroke and if the results is 1 means there ML model predicts a stroke. The results are: ', y_predicted_from_loaded_model) 



 '''