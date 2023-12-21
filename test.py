from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

File_path = 'D:/plugkub/data/'
File_name = 'car_data.csv'

df = pd.read_csv(File_path + File_name)
df.drop(columns=['User ID'], inplace=True)
df['Age'].fillna(method='pad', inplace=True)
df['AnnualSalary'].fillna(method='pad', inplace=True)

x = df.iloc[:, 1:3]
y = df['Purchased']
x_train, x_test, y_train, y_test= train_test_split(x, y, random_state=0)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)

feature_names = x.columns.tolist()
class_name = y.unique().tolist()

feature_imp = model.feature_importances_
feature_names = x.columns.tolist()

accuracy = model.score(x_train,y_train)
print('accuracytrain: {:.2f}'.format(accuracy))

accuracy = model.score(x_test,y_test)
print('accuracytest: {:.2f}'.format(accuracy))

sns.set(rc={'figure.figsize': (7.5, 5.22)})
sns.barplot(x=feature_imp, y=feature_names)

plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names=feature_names,
              class_names=class_name,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=18)
plt.show()
