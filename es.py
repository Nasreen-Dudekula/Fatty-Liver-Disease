import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tkinter import * 
from tkinter import messagebox
import os

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

# Naive bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
Y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_test, y_test) * 100, 2)

os.system('Keras 118 e '+str(acc_gaussian))

rez = ''
# Using readlines()
file1 = open('output/estats.txt', 'r')
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
