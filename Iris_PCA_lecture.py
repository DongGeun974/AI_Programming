from sklearn import datasets
import matplotlib.pyplot as plt

## 데이터 불러오기 및 확인 ##
iris = datasets.load_iris()
print(iris)

## Iris 원본 데이터 시각화 ##
print("Targets: " + str(iris.target_names))
print("Features: " + str(iris.feature_names))
print(iris.data[0:10,:])

## 4차원 데이터이기 때문에 2차원으로 시각화하기 위해 여러개의 그래프 창이 필요함. ##
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

plt.scatter(iris.data[:,0], iris.data[:,3], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

plt.scatter(iris.data[:,1], iris.data[:,2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

plt.scatter(iris.data[:,2], iris.data[:,3], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

## PCA 실행  및 new 데이터 생성 ##
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
new_data = pca.fit_transform(iris.data)
print(new_data)

## new 데이터 시각화 ##
plt.scatter(new_data[:,0], new_data[:,1], c=iris.target)
plt.xlabel('Principal Component 1 (1st dimension)')
plt.ylabel('Principal Component 2 (2nd dimension)')
plt.show()

## new 데이터 3차원 시각화 ##
pca3 = PCA(n_components=3)
new_data = pca3.fit_transform(iris.data)
fig = plt.figure()
from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(new_data[:,0], new_data[:,1], new_data[:,2], c=iris.target)
plt.show()