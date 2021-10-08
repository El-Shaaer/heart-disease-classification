##################
#Import libraries#
##################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
#import metrics here
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


class HeartDiseaseClassifier:

    def __init__(self, path):
        """Load DataFrame"""
        self.df = pd.read_csv(path)
        # print(self.df.describe())
        # print(self.df)

    def column_names(self):
        """the function returns the name of all columns"""
        columns = list(self.df.columns.drop("HeartDisease"))
        return columns

    def correlations(self):
        """Dsiplay correlation using heatmap or pairplot with all features in the DataFrame"""
        type = input("Choose a type of visualization 'corr', or 'pair': ")
        if type == "corr":
            sns.heatmap(self.df.corr(), annot=True)
        elif type == "pair":
            sns.pairplot(self.df)
        plt.show()
        

    
    def convert_columns(self):
        """Recall previous functions to convert the three features 1-Sex 2-ExerciseAngina and 3-ST_Slope"""
        self.df = self.df.apply(LabelEncoder().fit_transform)
        
        
    def delete_columns(self):
         """Delete unncessary columns"""
         self.df.drop(["RestingECG"], inplace=True, axis=1)
        
    def split_data(self):
         """Split DataFrame into X -> features and y -> the target
         the function returns X_train, X_test, y_train, y_test"""
         X = self.df.drop("HeartDisease", axis = 1)
         y = self.df["HeartDisease"]
         X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=3)
         return X_train, X_test, y_train, y_test
    
    def split_display_shapes(self, X_train, X_test, y_train, y_test):
        """Display the shapes of training and testing"""
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        
    def regression_model(self,X_train, y_train):
         """Evaluate and fit the DecisionTreeRegressor model by x_train, y_train
    #     the function returns the fitted model variable that can be used for predictions later"""
         model = RandomForestClassifier()
         fitted_model = model.fit(X_train, y_train)
         return fitted_model
    
    def model_accuracy_score(self,model, X_train, X_test, y_train, y_test):
         """How accurate is RandomForestClassifier model on training and testing"""
         print("Model train accuracy score :",model.score(X_train, y_train))
         print("Model test accuracy score :",model.score(X_test, y_test))
        
    def save_model(self, model, name):
         """Save the fitted model to deploy it and use it with it`s weight later"""
         joblib.dump(model,name + ".h5")
         print("Model saved sucessfuully")
        
   
    
    

class HeartDiseasePredictor:
    def __init__(self, model_name):
        self.name= model_name
        
        
    
    def loadmodel(self):
        """Load the model that has been saved already"""
        self.model = joblib.load(self.name + ".h5")
        print(f"The model {self.model} has been loaded sucessfully")
        
     
         
    def predict(self):
        
        lst = [] 
        #number of elements as input
        n = int(input("Enter how many analysis result you will input : "))
        # iterating till the range
        features = ["Age: ", "Sex(0 or 1): ", "Chest Pain Type: ", "Resting BP: ", "Cholesterol: ", "Fasting BS: ", "Max HR: ", "Exercise Angina: ", "Oldpeak: ", "ST Slope: "]
          
        for i in range(0, n):
            ele = int(input(features[i]))
            lst.append(ele)     

        result= self.model.predict([lst])[0]
        if result == 1:
            print("Heart Disease, Check Doctor")
        else: 
            print("Normal keep Healthy ")
        

    
if __name__ == "__main__":
    
    #1- load DataFrame
    classifier = HeartDiseaseClassifier("heart.csv")
    #Correlation of the Ddataset before modified
    classifier.correlations()
    #2- convert object data (i.e categorical data) into numerical data.
    classifier.convert_columns()
    #3- delete columns
    classifier.delete_columns()
    print(classifier.df)
    #4-split data into training and testing
    X_train, X_test, y_train, y_test = classifier.split_data()
    #5-display the shape of training and testing
    classifier.split_display_shapes(X_train, X_test, y_train, y_test)
    print("--"*18)
    # #7- excute and train the decision tree model
    random_forest = classifier.regression_model(X_train, y_train)
    # #8- predict the y_testing using x_testing...
    y_prediction = random_forest.predict(X_test)
    # #9- display: How accuate is the model on training and testing
    classifier.model_accuracy_score(random_forest, X_train, X_test, y_train, y_test)
    print("--"*18)
    #11- Finally: save and load model
    classifier.save_model(random_forest, input("Type in the name of the file: "))
    
    # Initializing the predictor class
    predector= HeartDiseasePredictor(input('model name: '))
    # load the model
    predector.loadmodel()
    # predict given user input
    predector.predict()
    
    # #End project

    
    
    
        
    
    
    
    
        
    






