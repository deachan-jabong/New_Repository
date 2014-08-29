##Titanic Survival Prediction


###CONTENTS
* Data Cleaning
* Exploring Data through Visualizations with Matplotlib
* Classification
* Filtering the data using k-means clustering
* Creating a template for a submission to Kaggle

###DATA CLEANING:
<ol>
<li>Involves getting rid of silly values, filling in missing values, discretising numeric attributes. </li>
<li>Age attribute,ticket number and Cabin are seen to have the maximum number of blank spaces.</li>
<li>Age is a great deciding factor in the decision about the survival or non-survival of the records. This is seen using the visualization of data which plots Age vs Survival.</li>
<li>Ticket number and Cabin don't seem to add much information to the classification, as the fare and Pclass are enough to clearly tell about the passenger's class, so cabin and ticket number can be dropped safely.</li>
<li>Age column was refined initially using, filling the means of Age and means of Fare attribute in the missing age and fare feilds repectively.</li>
<li>Filling mode values of Embarked value in the missing Embarked attribute of the records
          
<pre><code>
    #cleaning the Fare column
    train.Fare = train.Fare.map(lambda x: np.nan if x==0 else x)
    classmeans = train.pivot_table('Fare', rows='Pclass', aggfunc='mean')
    train.Fare = train[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )
    
    #cleaning the age column
    meanAge=np.mean(train.Age)
    train.Age=train.Age.fillna(meanAge)
    
    #cleaning the embarked column
    modeEmbarked = mode(train.Embarked)[0][0]
    train.embarked = train.Embarked.fillna(modeEmbarked)
</pre></code></li>
    
<li>After further analysis of the data, the age feild was filled as per:
<ol><li>If the record's name is tagged as Mr or Mrs, the age attribute is filled with the average age of the adults i.e. around 38-40.</li>
<li>If the record's name is tagged as Miss or Master,the age attribute is filled with the average age of the younger generation which is 15.</li></ol>
</li></ol>

#####EXPLORING DATA THROUGH VISUALIZATIONS WITH MATPLOTLIB:
<pre><code>
fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(train['Age'], bins = 10, range = (train['Age'].min(),train['Age'].max()))
plt.pyplot.title('Age distribution')
plt.pyplot.xlabel('Age')
plt.pyplot.ylabel('Count of Passengers')
plt.pyplot.show()

fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(train['Fare'], bins = 10, range = (train['Fare'].min(),train['Fare'].max()))
plt.pyplot.title('Fare distribution')
plt.pyplot.xlabel('Fare')
plt.pyplot.ylabel('Count of Passengers')
plt.pyplot.show()

temp1 = train.groupby('Pclass').Survived.count()
temp2 = train.groupby('Pclass').Survived.sum()/train.groupby('Pclass').Survived.count()
fig = plt.pyplot.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers by Pclass")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Pclass')
ax2.set_ylabel('Probability of Survival')
ax2.set_title("Probability of survival by class")
</pre></code>

Further data cleaning and addition of few more relevant columns after analysis and removal of redundant columns like name,sex and age:

<pre><code>def woman_child_or_man(passenger):
        Age, Sex = passenger
        if Age<16:
                return 1
        elif Sex == 0:
                return 40
        else:
                return 2
                
def Age_bucket(passenger1):
        Age,Sex = passenger1
        y = Sex
        if Age<16:
                return 1        
        elif Age<55:
                return 3
        elif Age<70:
                return 4
        else:
                return 5
                
def Fare_category(Passenger2):
       Fare,Pclass = Passenger2
       if Fare <= 11:
                return 1
       elif 11 < Fare and Fare <= 30:         
                return 2         
       else:
                if Pclass==1:
                        return 4
                return 3

def main():
    #read in the training file
    print '\nRandom Forest And Adaboost Classifier Starts!\n'
    train = pd.read_csv("/home/deachan/Desktop/data/set5/Train.csv")
    
    #set the training responses
    target = train[train.columns[1]]
    
    #set the training features
    train["who"] = train[["Age", "Sex"]].apply(woman_child_or_man, axis=1)
    train["adult_male"] = (train.who == 40).astype(int)
    train["alone"] = (~(train.Parch + train.SibSp).astype(bool)).astype(int)
    train["Age1"] = train[["Age", "Sex"]].apply(Age_bucket,axis=1)
    train["Fare_cat"] = train[["Fare","Pclass"]].apply(Fare_category,axis=1)
    train = train.drop(["Name","Age","Sex"], axis=1)
</pre></code>
 
###CLASSIFICATION

Methods Used                    | Acuracy             | Shortcomings                   |Reasons for Selection            |
--------------------------------|---------------------|--------------------------------|------------------------------------------------------|
SVM with linear kernel          | 71 percent          |   Slow  |                                                           |
--------------------------------|---------------------|--------------------------------|-------------------------------------------------------|
SVM with polynomial kernel(Best degree) | 73 percent | Slow  |The hyperplane for classification maynot be linear.  |
                   |                     |                                |                                                           |
--------------------------------|---------------------|--------------------------------|-------------------------------------------------------|
SVM with Gaussian kernel  | 75 percent          | Slow |The hyperplane for classification maynot be linear.Outliers dealt more accurately |
--------------------------------|---------------------|--------------------------------|-------------------------------------------------------|
SVM with Gaussian kernel(gamma=0.0001 and box constraint C=100000)   | 77 percent  |Slow |The gaussian kernel selection proved to be the best as the classification hyperplane maynot be linear. |
--------------------------------|---------------------|--------------------------------|--------------------------------------------------------|
Random Forest Classifier |75 percent |                                |Constructs a multitude of decsion trees and often believed to give the best for high number of estimators(more trees) results.Gave good accuracy |
--------------------------------|---------------------|--------------------------------|-----------------------------------------------------------|
Adaboost Classifier   |75 percent |                                | Gave good enough accuracy |
--------------------------------|---------------------|--------------------------------|---------------------------------------------------------|
K-nearest neighbour Classifier  |71 percent           |Low on accuracy                 |Nearest Distance classification,more like clustering       |
--------------------------------|---------------------|--------------------------------|---------------------------------------------------------|
Linear regression               |72 percent           |Low on accuracy                 |                                                           |
--------------------------------|---------------------|--------------------------------|---------------------------------------------------------|
Logistic Regression             |71 percent           |                                |Bayesian Ridge gave good accuracy compared to rest of the logistic regression methods |


#####FINAL MODEL:
Classification using combination of 3 most efficient classifiers i.e. svm.SVC, random forest classifier and adaboost (78.458 Percent accuracy)
<pre><code>rf0 = AdaBoostClassifier(n_estimators=200,algorithm='SAMME')
    rf1 = RandomForestClassifier(n_estimators=1000, max_features = None, n_jobs=-1)
    rf2 = svm.SVC(kernel='rbf',C = C1,cache_size = 200, gamma = ga)
    
    #Fitting the model
    print "\n***********************************************************************\n"
    print('Fitting the model\n')
    rf0.fit(train, target)
    rf1.fit(train, target)
    rf2.fit(train, target)
    rf3.fit(train, target)
    
    # run model against test data
    predicted_probs0 = rf0.predict(realtest)  
    predicted_probs1 = rf1.predict(realtest)
    predicted_probs2 = rf2.predict(realtest)
    predicted_probs3 = rf3.predict(realtest)
    
    #Store the dataframe as np.array
    predicted_probs0 = ["%d" % x for x in predicted_probs0]
    predicted_probs1 = ["%d" % x for x in predicted_probs1]
    predicted_probs2 = ["%d" % x for x in predicted_probs2]
    predicted_probs3 = ["%d" % x for x in predicted_probs3]
       
    total = len(realtest)
    Matrix = [[0 for x in xrange(total)] for x in xrange(4)]
    count = [0 for x in xrange(total)]
    for i in range(4):
        for j in range(total):
                y = "{}{}".format("predicted_probs", i)
                Matrix[i][j] = eval(y)[j]
                count[j] = count[j] + int(Matrix[i][j])

   for jj in range(total):
        if count[jj] > 2:
                Matrix[3][jj]  = 1
        else:
                Matrix[3][jj] = 0
    predicted_probs["Survived"] = [0 for x in range(len(realtest))]
    predicted_probs["Survived"] = Matrix[3][:]
</pre></code>

###FILTERING THE DATA USING CLUSTERING TECHNIQUE, K-MEANS:
<ol>
<li>Training a k-means clustering model using the training set which gives back centroids of the clusters formed.</li>
<li>Predicting the clusters to which the records in test set belongs to, using the minimum eucledian distance from these cluster centroids.
<pre><code>centers, idx = kmeans2(np.array(zip(np.array(train.Pclass),np.array(train.Fare),np.array(train.who),np.array(train.alone))),3)
        train["Cluster"] = idx
        for k in range(len(realtest)): 
        for j in range(len(centers)):
                x = realtest.Pclass[k] - centers[j][0]
                y = realtest.Fare[k] - centers[j][1]
                a = realtest.alone[k] - centers[j][3]
                z = realtest.who[k] - centers[j][2]
                
                distance[j] =np.abs(np.sqrt((x**2)+(z**2)+(a**2)+(y**2))) #+(a**2)+(y**2)
                
        realtest.Cluster[k] = distance.index(min(distance))
</pre></code></li>
                
<li>
Another Method: Expectation Maximization: 
<ol>
<li>Used by the inbuilt method gaussian mixture model(GMM).</li>
<li>Expected to give the best fit model but in this particular case, wasn't giving any better result than k-means.</li>
<li>Reason for trying expectation maximization was that in case of k-means, the way to initialize the means was not specified, the k samples are chosen randomly. Therefore, the results produced depend on the initial values for the means, and it frequently happens that suboptimal partitions are found.The results depend on the value of k.</li>
<li>Expectation Maximization maximizes the likelihood of a sample being assigned to a particular cluster.</li>
</ol></li>
<li>Additional filter used using the Bayesian Ridge logistic regression model to filter the data.Contributed in bringing up the accuracy from 79.458 to 79.904 percent</li></ol>
        
        
###Creating a template for a submission to Kaggle.
<pre><code>
    predicted_probs = pd.DataFrame(realtest["PassengerId"])
    predicted_probs["Survived"] = [0 for x in range(len(realtest))]
    predicted_probs["Survived"] = Matrix[3][:]
    df = pd.DataFrame(predicted_probs)
    df.to_csv("/home/deachan/Desktop/data/set5/Newest8.csv", index = False, sep=',', encoding='utf-8')
</pre></code>
