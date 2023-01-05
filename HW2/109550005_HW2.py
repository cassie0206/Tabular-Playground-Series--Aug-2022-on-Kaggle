import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# 1. Compute the mean vectors mi, (i=1,2) of each 2 classes
# m1 => label=0, m2 => label=1
# extract the label belonging to the class0's data to class0
train_c1 = x_train[np.where(y_train == 0)]
train_c2 = x_train[np.where(y_train == 1)]

# take average
m1 = np.mean(train_c1, axis=0)
m2 = np.mean(train_c2, axis=0)

print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")

# 2. Compute the Within-class scatter matrix SW
# according to the formula in textbook
sw = np.zeros((2, 2))
# print(np.sum(np.dot((train_c1 - m1), (train_c1 - m1).T)), axis=0)
sw += np.dot((train_c1 - m1).T, (train_c1 - m1))
sw += np.dot((train_c2 - m2).T, (train_c2 - m2))


print(f"Within-class scatter matrix SW: {sw}")

# 3. Compute the Between-class scatter matrix SB
# m is the overall average
# according to the formula in the textbook
sb = np.zeros((2, 2))
m1 = m1.reshape(m1.shape[0], 1)
m2 = m2.reshape(m2.shape[0], 1)

sb = np.dot((m2 - m1), (m2 - m1).T)

print(f"Between-class scatter matrix SB: {sb}")

# 4. Compute the Fisher’s linear discriminant
w = np.dot(np.linalg.inv(sw), (m2 -m1))

print(f" Fisher’s linear discriminant: {w}")

# 5. Project the test data by linear discriminant and get the class prediction 
# by K nearest-neighbor rule. Please report the accuracy score with K values from 1 to 5

train_project = np.dot(x_train, w) 
train_project = train_project.flatten() 
# print(train_project)
test_project = np.dot(x_test, w)
test_project = test_project.flatten()

# KNN => k=1~5
for k in range(1, 6):
    y_pred = list()
    for i in test_project:
        dis = list()
        for index, j in enumerate(train_project):
            dis.append([(i - j) ** 2, y_train[index]])
        dis.sort()
        c1 = c2 = 0
        for v in dis[:k]:
            if v[1] == 0:
                c1 += 1
            else:
                c2 += 1
        if c1 >= c2:
            y_pred.append(0)
        else:
            y_pred.append(1)
    acc = accuracy_score(y_test, y_pred)
    print('k=',k)
    print(f"Accuracy of test-set {acc}")


# 6. Plot the 1) best projection line on the training data and show the slope and intercept 
# on the title (you can choose any value of intercept for better visualization) 
# 2) colorize the data with each class 3) project all data points on your projection line. 
slope = w[1] / w[0]

# plot training data
for p in train_c1:
    plt.scatter(p[0], p[1], c='blue', s=5)
for p in train_c2:
    plt.scatter(p[0], p[1], c='red', s=5)

# plot projection line
X = np.linspace(-4, 2, 10)
y = slope * X 
plt.plot(X, y, lw=1, c='c')

# Plot projection points
# print('test: ', (train_project * w /(np.sum(w ** 2)))[0])
# print('test1: ', (x_train @ w.reshape(w.shape[0], 1) / np.sum(w ** 2) * w.reshape(1, w.shape[0])).T[0])
train_project_0 = (train_project * w /(np.sum(w ** 2)))[0] 
train_project_1 = (train_project * w /(np.sum(w ** 2)))[1] 

plt.scatter(train_project_0[np.where(y_train == 0)], train_project_1[np.where(y_train == 0)], c='b', s=5)
plt.scatter(train_project_0[np.where(y_train == 1)], train_project_1[np.where(y_train == 1)], c='r', s=5)

# Plot line of data point and projection points
for i in range(len(train_c1)):
    plt.plot([train_c1[i][0], train_project_0[np.where(y_train == 0)][i]], [train_c1[i][1], train_project_1[np.where(y_train == 0)][i]], c='c', alpha=0.05)
for i in range(len(train_c2)):
    plt.plot([train_c2[i][0], train_project_0[np.where(y_train == 1)][i]], [train_c2[i][1], train_project_1[np.where(y_train == 1)][i]], c='c', alpha=0.05)

plt.title(f'Projection Line: w= %.5f , b= 0' % slope)
plt.show()