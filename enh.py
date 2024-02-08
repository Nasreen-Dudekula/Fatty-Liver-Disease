import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import itertools
from tkinter import * 
from tkinter import messagebox
import os

my_listLabels = []
my_list = []

hd = pd.read_csv("flp_ds.csv", sep = ",")

print(hd.shape)
print(hd.info())
print(hd.head())


total = hd.isnull().sum().sort_values(ascending =False)
percent = (hd.isnull().sum()/hd.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])

# Split train and test data
x = hd.drop("flp", axis=1)
y = hd["flp"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state=42)

#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print('\nTraining Size : ')
print(y_train.shape)
print('\nTesting Size : ')
print(y_test.shape)

# Xgboost Classifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
xg = XGBClassifier()
xg.fit(x_train, y_train)
y_pred = xg.predict(x_test)
acc_xg = round(xg.score(x_test, y_test) * 100, 2)

os.system('Keras 118 ann '+str(acc_xg))

rez = ''
# Using readlines()
file1 = open('output/xstats.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    #print("Line{}: {}".format(count, line.strip()))
    rez = rez + line.strip()+'\n';


root = Tk()
root.geometry("300x200")  
w = Label(root, text =rez, font = "50") 
w.pack()
root.mainloop()

p1 = []
p2 = []
p3 = []
p4 = []
p5 = []

with open("output/estats.txt", encoding="utf-8") as f:
    cnt = 0
    for line in f:
        cnt = cnt+1
        result = [line.strip() for line in line.split(':')]
        if cnt == 1:            
            p1.append(float(result[1]))
        if cnt == 2:            
            p2.append(float(result[1]))
        if cnt == 3:            
            p3.append(float(result[1]))
        if cnt == 4:            
            p4.append(float(result[1]))
        if cnt == 5:            
            p5.append(float(result[1]))        
        
with open("output/pstats.txt", encoding="utf-8") as f:
    cnt = 0
    for line in f:
        cnt = cnt+1
        result = [line.strip() for line in line.split(':')]
        if cnt == 1:            
            p1.append(float(result[1]))
        if cnt == 2:            
            p2.append(float(result[1]))
        if cnt == 3:            
            p3.append(float(result[1]))
        if cnt == 4:            
            p4.append(float(result[1]))
        if cnt == 5:            
            p5.append(float(result[1]))        

with open("output/xstats.txt", encoding="utf-8") as f:
    cnt = 0
    for line in f:
        cnt = cnt+1
        result = [line.strip() for line in line.split(':')]
        if cnt == 1:            
            p1.append(float(result[1]))
        if cnt == 2:            
            p2.append(float(result[1]))
        if cnt == 3:            
            p3.append(float(result[1]))
        if cnt == 4:            
            p4.append(float(result[1]))
        if cnt == 5:            
            p5.append(float(result[1]))        


###########################

my_listLabels = []
my_list = []

my_listLabels.append('NB')            
my_list.append(p1[0])
my_listLabels.append('RF')            
my_list.append(p1[1])
my_listLabels.append('Hyb(ANN+XGB)')            
my_list.append(p1[2])

# Plot the bar graph
plot = plt.bar(my_listLabels,my_list)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()    
    plt.text(value.get_x() + value.get_width()/2.,1.002*height,'%f' % float(height), ha='center', va='bottom')
 
# Add labels and title
plt.title("Fatty Liver Prediction Accuracy")
plt.xlabel("Classifier")
plt.ylabel("Score")
 
# Display the graph on the screen
plt.show()

#############################

###########################

my_listLabels = []
my_list = []

my_listLabels.append('NB')            
my_list.append(p2[0])
my_listLabels.append('RF')            
my_list.append(p2[1])
my_listLabels.append('Hyb(ANN+XGB)')            
my_list.append(p2[2])

# Plot the bar graph
plot = plt.bar(my_listLabels,my_list)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()    
    plt.text(value.get_x() + value.get_width()/2.,1.002*height,'%f' % float(height), ha='center', va='bottom')
 
# Add labels and title
plt.title("Fatty Liver Prediction Sensivity")
plt.xlabel("Classifier")
plt.ylabel("Score")
 
# Display the graph on the screen
plt.show()

#############################

###########################

my_listLabels = []
my_list = []

my_listLabels.append('NB')            
my_list.append(p3[0])
my_listLabels.append('RF')            
my_list.append(p3[1])
my_listLabels.append('Hyb(ANN+XGB)')            
my_list.append(p3[2])

# Plot the bar graph
plot = plt.bar(my_listLabels,my_list)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()    
    plt.text(value.get_x() + value.get_width()/2.,1.002*height,'%f' % float(height), ha='center', va='bottom')
 
# Add labels and title
plt.title("Fatty Liver Prediction Specificity")
plt.xlabel("Classifier")
plt.ylabel("Score")
 
# Display the graph on the screen
plt.show()

#############################

###########################

my_listLabels = []
my_list = []

my_listLabels.append('NB')            
my_list.append(p4[0])
my_listLabels.append('RF')            
my_list.append(p4[1])
my_listLabels.append('Hyb(ANN+XGB)')            
my_list.append(p4[2])

# Plot the bar graph
plot = plt.bar(my_listLabels,my_list)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()    
    plt.text(value.get_x() + value.get_width()/2.,1.002*height,'%f' % float(height), ha='center', va='bottom')
 
# Add labels and title
plt.title("Fatty Liver Prediction AUROC")
plt.xlabel("Classifier")
plt.ylabel("Score")
 
# Display the graph on the screen
plt.show()

#############################
