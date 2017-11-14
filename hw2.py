
import pandas as pd
import numpy as np
from sklearn import tree, preprocessing
import graphviz

file = "bank.csv"
df = pd.read_csv(file, header = 0, sep = ';')
#data = np.loadtxt(fname = file, delimiter=',')

jobArray = np.array(df['job'].data)
le = preprocessing.LabelEncoder()
print(jobArray)
le.fit(jobArray)
print(le.classes_)
transformedJobs = list(le.transform(jobArray))
for i in range(len(transformedJobs)):
    print(str(transformedJobs[i]) + "    " + jobArray[i])

print("------------------")

ageData = list(df['age'].data)
df1 = []
for i in range(len(ageData)):
    df1.append([ageData[i], transformedJobs[i]])

df_y = list(np.array((df['y'].data)))
print(len(df1))
print(len(df_y))
clf = tree.DecisionTreeClassifier()
clf = clf.fit(df1, df_y)

dot_data  = tree.export_graphviz(clf, out_file=None, class_names=['Yes', 'No'])
graph = graphviz.Source(dot_data)
graph.render("age")


