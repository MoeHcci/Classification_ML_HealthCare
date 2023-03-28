
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

