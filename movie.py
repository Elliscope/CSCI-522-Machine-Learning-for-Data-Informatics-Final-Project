# import load_iris function from datasets module
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.metrics import f1_score
import csv
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.cluster import KMeans # KMeans clustering 
import matplotlib.pyplot as plt # Python defacto plotting library
import seaborn as sns # More snazzy plotting library
from gensim.models import word2vec
from sklearn import svm
from sklearn import ensemble
from sklearn.feature_selection import RFE
from sklearn.datasets import make_friedman1

#Load csv data
with open("movie_metadata.csv", 'rb') as f:
    raw_data = list(csv.reader(f))

#clean the dataset by ignoring it if any column is empty or equal to 0

data = []
counter = 0
for row in raw_data:
    qualified = True
    for column in row:
        if column == " "or column == 0 or column =="0":
            qualified = False
    if qualified:
        data.append(row)
    counter = counter + 1

#the size of the current dataset right now
print len(data)

movie = pd.read_csv('movie_metadata.csv') # reads the csv and creates the dataframe called movie
movie.head()

#plot_keywords -> numerical values
#dictioinarize the plot keywords section with their corresponding imdb_score
plot_keywords_dic = {}
for index, row in movie.iterrows():
    words_array = str(row["plot_keywords"]).split("|")
    for word in words_array:
        if word not in plot_keywords_dic:
            plot_keywords_dic[word] = []
        plot_keywords_dic[word].append(row["imdb_score"])

#compute the average of all the plot keywords in the dictionary
for key, elem in plot_keywords_dic.items():
    plot_keywords_dic[key] = sum(elem)/len(elem)

        
#convert the orignal movie value to the sum of the top five keywords foundable in the training dataset
for index, row in movie.iterrows():
    words_array = str(row["plot_keywords"]).split("|")
    plot_value = 0
    counter = 0
    i = 0
    while counter < 5 and i < len(words_array):
        if(words_array[i] in plot_keywords_dic):
            plot_value = plot_value + plot_keywords_dic[words_array[i]]
            counter = counter + 1
        i = i + 1
    movie.set_value(index,'plot_keywords',plot_value)
    

#Genres -> Column Feature
genre_list = []
for index, row in movie.iterrows():
    words_array = str(row["genres"]).split("|")
    for word in words_array:
        if word not in genre_list:
            genre_list.append(word)
            
for movie_genre in genre_list:
    movie[movie_genre] = pd.Series(0,index=movie.index)
    
#convert the orignal movie value to the sum of the top five keywords foundable in the training dataset
for index, row in movie.iterrows():
    words_array = str(row["genres"]).split("|")
    for word in words_array:
        movie.set_value(index,word,1)
    

#Country -> Index
country_dic = {}
for index, row in movie.iterrows():
    word = str(row["country"])
    if word not in country_dic:
        country_dic[word] = [1]
        
country_index = 1
#assign each country an index number
for key, elem in country_dic.items():
    country_dic[key] = country_index
    country_index = country_index+1
    
for index, row in movie.iterrows():
    word = str(row["country"])
    movie.set_value(index,'country',country_dic[word])



# converting director_name, actor1_name, actor2_name, actor3_name, content rating, and language to numeric values
director_map = {}
count = 0
for name in movie["director_name"]:
    if not director_map.has_key(str(name)):
        #dict[str(name)] = count
        director_map.update({str(name): count})
        count+=1
#print director_map

#for index, row in movie_data.iterrows():
#    if not str(row['director_name']) in director_map:
#        director_map.update({str(row['director_name']): len(director_map)})
#    movie_data.set_value(index, 'director_name', director_map[str(row['director_name'])])
        
# converting actor1 to values
actors_map = {}
actors_count = 0

tempActors = movie["actor_1_name"].copy(True);
pd.np.random.shuffle(tempActors)
#for name in movie["actor_1_name"]:
for name in tempActors:
    #print name
    if not actors_map.has_key(str(name)):
        actors_map.update({str(name): actors_count})
        actors_count+=1

# converting actor2 to values
tempActors = movie["actor_2_name"].copy(True);
pd.np.random.shuffle(tempActors)
#for name in movie["actor_2_name"]:
for name in tempActors:
    if not actors_map.has_key(str(name)):
        actors_map.update({str(name): actors_count})
        actors_count+=1
        
tempActors = movie["actor_3_name"].copy(True);
pd.np.random.shuffle(tempActors)
# converting actor3 to values
#for name in movie["actor_3_name"]:
for name in tempActors:
    if not actors_map.has_key(str(name)):
        actors_map.update({str(name): actors_count})
        actors_count+=1
        
rating_map = {}
rating_count = 0
for rating in movie["content_rating"]:
    if not str(rating) in rating_map:
        rating_map.update({str(rating): rating_count})
        rating_count+=1
        #print rating
        
language_map = {}
language_count = 0
for language in movie["language"]:
    if not str(language) in language_map:
        language_map.update({str(language): language_count})
        language_count+=1
        
for index, row in movie.iterrows():
    movie.set_value(index, 'director_name', director_map[str(row['director_name'])])
    movie.set_value(index, 'actor_1_name', actors_map[str(row['actor_1_name'])])
    movie.set_value(index, 'actor_2_name', actors_map[str(row['actor_2_name'])])
    movie.set_value(index, 'actor_3_name', actors_map[str(row['actor_3_name'])])    
    movie.set_value(index, 'language', language_map[str(row['language'])])
    movie.set_value(index, 'content_rating', rating_map[str(row['content_rating'])])
    
movie.head()

str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in movie.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
    
#include the column name if applicable 
str_list.append("imdb_score")
            

# Get to the numeric columns by inversion            
num_list = movie.columns.difference(str_list)  

movie_num = movie[num_list]
#del movie # Get rid of movie df as we won't need it now
#movie_num.head()

#Remove some existing features generated after the movie is screening
del movie_num['gross']
del movie_num['num_critic_for_reviews']
del movie_num['num_user_for_reviews']

movie_num = movie_num.fillna(value=0, axis=1)
X = movie_num.values
#print movie_num.head()
# Data Normalization
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 8))
plt.title('Pearson Correlation of Movie Features')
# Draw the heatmap using seaborn
sns.heatmap(movie_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black')

#Pearson Correlation shows how does each variable corelate to each other
#1 means totally linear correlation, 0 means no correlation and -1 means totally negative linear correlation

#explained variance measure
# Calculating Eigenvectors and eigenvalues of Cov matirx
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)


# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
#print cum_var_exp

# make 5 folds
print len(movie_num)

# there are 5043 samples, so we will have 1008 in 4 training sets, and 1011 in the test set
x_folds = []
y_folds = []
x_folds.append(movie_num[0:1007].values.tolist())
y_folds.append(movie["imdb_score"][0:1007].values.T.tolist())
x_folds.append(movie_num[1008:2015].values.tolist())
y_folds.append(movie["imdb_score"][1008:2015].values.T.tolist())
x_folds.append(movie_num[2016:3023].values.tolist())
y_folds.append(movie["imdb_score"][2016:3023].values.T.tolist())
x_folds.append(movie_num[3024:4031].values.tolist())
y_folds.append(movie["imdb_score"][3024:4031].values.T.tolist())
x_folds.append(movie_num[4032:5042].values.tolist())
y_folds.append(movie["imdb_score"][4032:5042].values.T.tolist())

# set the test set to be the 5th fold
x_test_set = x_folds[4]
y_test_set = y_folds[4]

# now merge the 4 folds in 4 different ways for cross validation
X_training_sets = []
Y_training_sets = []

X_validation_sets = []
Y_validation_sets = []

# merge folds 0,1,2, leave fold 3 for validation
X_training_sets.append([])
Y_training_sets.append([])
X_training_sets[0].extend(x_folds[0])
Y_training_sets[0].extend(y_folds[0])
X_training_sets[0].extend(x_folds[1])
Y_training_sets[0].extend(y_folds[1])
X_training_sets[0].extend(x_folds[2])
Y_training_sets[0].extend(y_folds[2])

X_validation_sets.append([])
X_validation_sets[0].extend(x_folds[3])
Y_validation_sets.append([])
Y_validation_sets[0].extend(y_folds[3])

# merge folds 1,2,3 leave fold 0 for validation
X_training_sets.append([])
Y_training_sets.append([])
X_training_sets[1].extend(x_folds[1])
Y_training_sets[1].extend(y_folds[1])
X_training_sets[1].extend(x_folds[2])
Y_training_sets[1].extend(y_folds[2])
X_training_sets[1].extend(x_folds[3])
Y_training_sets[1].extend(y_folds[3])

X_validation_sets.append([])
X_validation_sets[1].extend(x_folds[0])
Y_validation_sets.append([])
Y_validation_sets[1].extend(y_folds[0])

# merge folds 0,2,3 leave fold 1 for validation
X_training_sets.append([])
Y_training_sets.append([])
X_training_sets[2].extend(x_folds[0])
Y_training_sets[2].extend(y_folds[0])
X_training_sets[2].extend(x_folds[2])
Y_training_sets[2].extend(y_folds[2])
X_training_sets[2].extend(x_folds[3])
Y_training_sets[2].extend(y_folds[3])

X_validation_sets.append([])
X_validation_sets[2].extend(x_folds[1])
Y_validation_sets.append([])
Y_validation_sets[2].extend(y_folds[1])

# merge folds 0,1,3 leave fold 2 for validation
X_training_sets.append([])
Y_training_sets.append([])
X_training_sets[3].extend(x_folds[0])
Y_training_sets[3].extend(y_folds[0])
X_training_sets[3].extend(x_folds[1])
Y_training_sets[3].extend(y_folds[1])
X_training_sets[3].extend(x_folds[3])
Y_training_sets[3].extend(y_folds[3])

X_validation_sets.append([])
X_validation_sets[3].extend(x_folds[2])
Y_validation_sets.append([])
Y_validation_sets[3].extend(y_folds[2])

#print X_validation_sets


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

x_training_4folds = []
y_training_4folds = []


x_training_4folds.extend(np.array(movie_num[0:4031].values.tolist()))
y_training_4folds.extend(np.array(movie["imdb_score"][0:4031].values.T.tolist()))
# x_training_4folds.extend(movie_num[0:100])
# y_training_4folds.extend(movie["imdb_score"][0:100])

x_training_4folds = np.array(x_training_4folds)

print "Type of x_training_4folds",type(x_training_4folds)
print "Type of x_training_4folds[0]",type(x_training_4folds[0])


#print x_training_4folds
#print y_training_4folds

# clf = LinearRegression(normalize=True)
# print clf.get_params()
# parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
# grid_search = GridSearchCV(LinearRegression(),parameters, cv=4)
# grid_search.fit(x_training_4folds, y_training_4folds)

# #grid_search.best_params_
# #print "For the Logistic Regression: "
# print "Best parameter value is: ", grid_search.best_params_

# clf = LinearRegression(copy_X=True, normalize=True, n_jobs=1, fit_intercept=True)
# clf.fit(x_training_4folds, y_training_4folds)
# y_test_pred = clf.predict(x_test_set)





# parameters = {'C':[0.001, 0.01, 0.1, 1, 10]}
# grid_search = GridSearchCV(svm.SVR(),parameters, cv=4)
# grid_search.fit(x_training_4folds, y_training_4folds)

# print "Best parameter value is: ", grid_search.best_params_


print "SVR Model Results"


# X, y = make_friedman1(n_samples=50, n_features=64, random_state=0)
# print "Type of X",type(X)
# print "Type of X[0]",type(X[0])
# print X

column = list(movie_num)
array1 = [7,20,1,4,1,14,1,1,1,10,36,34,9,1,1,12,5,33,37,1,16,35,1,1,2,6,26,22,28,23,24,25,18,32,27,11,15,31,17,19,13,8,30,29,3,21]
array2 = [42,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,41,22,21,40,20,19,18,17,14,12,13,1,9,1,10,1,11,16,8,1,6,4,3,15,1,2,5,7]
array3 = [16, 21, 9, 31, 20, 18, 32, 12, 15, 17, 37, 33, 29, 26, 30, 27, 19, 36, 35, 22, 13, 34, 28, 23, 25, 24, 4, 1, 3, 1, 1, 2, 14, 1, 1, 6, 10, 5, 1, 1, 11, 8, 7, 1, 1, 1]

result1 = []
for iter in array2:
    if iter == 1:
        result1.append(column[iter])

print result1


# selector = RFE(clf, 5, step=1)
# selector = selector.fit(x_training_4folds, y_training_4folds)
# print selector.support_ 
# print selector.ranking_
# print len(selector.ranking_)
# clf = svm.SVR(kernel="linear",C=10)
# clf.fit(x_training_4folds, y_training_4folds)
# y_test_pred = clf.predict(x_test_set)
# mean_sq_error =  metrics.mean_squared_error(y_test_set, y_test_pred)

# print "Mean Squared Error: ", mean_sq_error
# print "Mean absolute error: ", metrics.mean_absolute_error(y_test_set, y_test_pred)
# print "Explained Variance: ", metrics.explained_variance_score(y_test_set, y_test_pred)
# print "R2: ", metrics.r2_score(y_test_set, y_test_pred)


# rfe = RFE(estimator=clf, n_features_to_select=1, step=1)
# print rfe.ranking

# print y_test_pred


# selector = RFE(clf, 1, step=1)
# selector = selector.fit(x_training_4folds, y_training_4folds)
# selector.support_ 
# selector.ranking_


# print "Mean Squared Error: ", metrics.mean_squared_error(y_test_set, y_test_pred)
# print "Mean absolute error: ", metrics.mean_absolute_error(y_test_set, y_test_pred)
# print "Explained Variance: ", metrics.explained_variance_score(y_test_set, y_test_pred)
# print "R2: ", metrics.r2_score(y_test_set, y_test_pred)