<h1>
  Classification Machine Learning (ML ) Project For Strokes Prediction
</h1>



<h2>General Information About The Project: </h2>


<ul>

<li>The whole project is available at healthcare.py file</li>
  <li>prediction.py is an example of inputting values and getting a prediction based on the ML models from the project</li>

  <li>The project investigates the <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset">Stroke Prediction
      Dataset</a> from kaggle.com and utilizes multiple ML Models based on the data</li>
  <li>The dataset is used to predict whether a patient is likely to get a stroke or not based 12 input feature
    parameters. 11 feature parameters if we ignore the "id" feature column </li>
  <li>The used ML models have the following constrains </li>
  <ul>
    <li>Patients 18+ </li>
    <li>Patients have a BMI below 60 </li>

  </ul>

  <li>The list of topics presented in this project are presented in their order below:</li>
  <ol>
    <li>Import all the main Python libraries</li>
    <li>Import the data</li>
    <li>Perform (Exploratory Data Analysis) EDA on the data</li>
    <ul>

      <li>Dropping columns if required </li>
      <li>Analyzing the unique inputs of each column </li>
      <li>Analyzing all the rows</li>
      <li>Conducting conclusions based on mathematical evidence</li>
      <li>Constructing plots based on the mathematical evidence</li>
    </ul>
    <li>Utilizing  OneHotEncoding, to ensure all of the categorical columns are ready for the ML process</li>
    <li>Splitting the data into a training dataset (70%) and a test dataset (30%)</li>
    <li>After conducting EDA, it came to realization that the data is unbalanced. Therefore, SMOTE was
        utilized to synthetically create random data for the unbalanced class and make the datasets
        balanced. SMOTE was applied to only the training dataset </li>
    <li>Scaling the data. Even though some of the employed ML models did not need scaling (e.g., Decision
        Trees and Random Forest) we preferred to use the scaled data to compare all the models based on the
        same exact scaled datasets</li>
    <li>Conduct the ML analysis by utilizing the following algorithms for classification</li>

    <ol>
        <li>Logistic Regression</li>
        <li>K-Nearest Neighbors (KNN)</li>
        <li>Support Vector Machines (SVM) For classification</li>
        <li>Decision Trees (With Adaboost and Gradient Boost)</li>
        <li>Random Forest</li>
    </ol>

</ol>

  </ul>
