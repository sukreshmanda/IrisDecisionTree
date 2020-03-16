from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree


data = load_iris()
index = [0,50,100]

#training data
training_data = np.delete(data.data,index,axis=0)
training_label = np.delete(data.target,index)

#testing data
test_data = data.data[index]
test_label = data.target[index]

#classifier 
cl = tree.DecisionTreeClassifier()
cl.fit(training_data,training_label)

print(data.target[test_label])
print(data.target[cl.predict(test_data)])


#visualize
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(cl,out_file=dot_data,
								feature_names=data.feature_names,
								class_names=data.target_names,
								filled=True,rounded=True,impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")
