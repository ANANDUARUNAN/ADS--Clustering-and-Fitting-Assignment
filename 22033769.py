
# Pandas is a tool for Python that lets you work with data sets. It has tools to help you analyse, clean, explore, and change data.
import pandas as PPD
# Numpy library is free to use and is used with almost every area of science and industry. 
import numpy as NPP
# importing the kmean from sk_learn package.
from sklearn.cluster import KMeans 
#Matplotlib is here for the data visualization which imported here as plotted 
import matplotlib.pyplot as plotted
#After importing matplotlib importing warnings to ignore the warnings which may come for future depreciation of methods name
import warnings 
warnings.filterwarnings('ignore') 

"""#creating the fucntion dataset which will read the dataset by pandas and skip four rows which is unneccessary and then droping
 some colum and setting index Country Name and and transposing then reseting index then returning two csv file."""
def Dataset(data_file):
    urban_population_data = PPD.read_csv(data_file, skiprows=4) 
    urban_population_data_1 = urban_population_data.drop(['Unnamed: 66', 'Indicator Code',  'Country Code'],axis=1) 
    urban_population_data_2 = urban_population_data_1.set_index("Country Name")  
    urban_population_data_2 = urban_population_data_2.T 
    urban_population_data_2.reset_index(inplace=True) 
    urban_population_data_2.rename(columns = {'index':'Year'}, inplace = True) 
    return urban_population_data_1, urban_population_data_2 

# define the path of data.
data_file = '/content/API_SP.URB.TOTL_DS2_en_csv_v2_5359282.csv'  
urban_population_Final_Dataset, urban_population_Transpose_data = Dataset(data_file)   
urban_population_Final_Dataset_5 = urban_population_Final_Dataset # showing starting rows. 
urban_population_Final_Dataset.head() 

urban_population_Transpose_data.head() #visualizing first five rows of data by the head function 
"""The function urban_population_Final_Dataset_2, which accepts the input urban_population_Final_Dataset, 
is defined in this code. The function gives urban_population_Final_Dataset_1 the input data as its input. 
Then, it uses the dropna() method to remove any rows with null values from the dataset, and it names the 
resulting dataframe urban_population_Final_Dataset_2. The function finally outputs urban_population_Final_Dataset_2.

The function is then defined, called, and given the argument urban_population_Final_Dataset. The returned
dataframe is then assigned to urban_population_Final_Dataset_3. Then, using the urban_population_Final_Dataset_3's
'Country Name' column, we access the country_name variable and store the value there. Finally, we use the head() method
to show the top 10 rows of urban_population_Final_Dataset_3."""

# Extracting 20 years of data with the help of function.
def urban_population_Final_Dataset_2(urban_population_Final_Dataset): 
    urban_population_Final_Dataset_1 = urban_population_Final_Dataset 
    urban_population_Final_Dataset_2 = urban_population_Final_Dataset_1.dropna() # drop null values from data.
    return urban_population_Final_Dataset_2

# calling the function to extract the data. 
urban_population_Final_Dataset_3 = urban_population_Final_Dataset_2(urban_population_Final_Dataset) 
country_name = urban_population_Final_Dataset_3['Country Name']
urban_population_Final_Dataset_3.head(10) # shows starting rows from data.

# check shape of data.
urban_population_Final_Dataset_3.shape 

# check null values from data.
urban_population_Final_Dataset_3.isnull().sum()
urban_population_Final_Dataset_3.describe().T 

# label coder via scikit learn is being brought in. 
from sklearn.preprocessing import LabelEncoder
# define predictor for encoder.
le_encod = LabelEncoder()
urban_population_Final_Dataset_3['Country Name'] = le_encod.fit_transform(urban_population_Final_Dataset_3['Country Name']) 
# showing 5 rows from data.
urban_population_Final_Dataset_3.head(10) 

X = urban_population_Final_Dataset_3.drop(['Country Name','Indicator Name'], axis=1)
y = urban_population_Final_Dataset_3['Country Name']  

# importing standard scaler for normalize the data.
from sklearn.preprocessing import StandardScaler
# define classifier.
stand_scaler = StandardScaler()
# fit classifier with data.  
stand_scaled = stand_scaler.fit_transform(X)
#Elbow Method To Findout Clusters.

"""We import the required libraries and specify the Cluster range, which establishes how many clusters the 
elbow function will take into account. To hold the mean distances for each cluster, we create an empty list 
called Meandist.

Then, using the KMeans class from scikit-learn, we loop through each value in the Cluster range and apply
K-means clustering. Using the cdist function from the scipy.spatial.distance module, the mean distance is 
determined.

Finally, we use the matplotlib.pyplot routines to plot the elbow curve. The figure size is set, the mean
distances are plotted against the number of clusters on the x-axis, the x and y axes are labelled, a grid 
is established, and the graph's title is provided. Finally, we use plt.show() to display the graph."""

# using the elbow method to find out the clusters.
from scipy.spatial.distance import cdist 
Cluster = range(10) 
Meandist = list()

for c in Cluster:
    algo = KMeans(n_clusters=c+1) 
    algo.fit(stand_scaled) 
    Meandist.append(sum(NPP.min(cdist(stand_scaled, algo.cluster_centers_, 'euclidean'), axis=1)) / stand_scaled.shape[0]) 

# Putting all the parameters and drawing the line.

# set figure size.
plotted.figure(figsize=(10,7))
# set parameter for graph.
plotted.plot(Cluster, Meandist, marker="o", color='y') 
# set xlabel.
plotted.xlabel('<----- Numbers of Clusters ----->',color='r', fontsize=15)
# set ylabel.
plotted.ylabel('<----- Average Distance ----->', color='r', fontsize=15) 
plotted.grid()
# set title for graph.
plotted.title('Choosing Clusters with Elbow Method', color='g', fontsize=20); 

# Set up the classification for grouping.
k_means_algo = KMeans(n_clusters=3, max_iter=100, n_init=10,random_state=10)
# fitting classifier with data.  
k_means_algo.fit(stand_scaled) 
# predict model to getting the label.
getting_predict = k_means_algo.predict(stand_scaled)  
getting_predict

"""This code defines a dictionary called diff_colors to map various cluster labels to various colours. 
A cluster label x is passed to the function colour, which returns the relevant colour from the diff_colors 
database.

Each label in k_means_algo.labels_ is subjected to the colour function using the map function, which produces 
a list of colours based on the cluster labels.

Then, using plt.scatter, a scatter plot is produced. The points' x and y coordinates are taken out of the
 dataframe X. In order to give each point a different colour depending on its cluster label, the c parameter is set to colours.

Using plt.xlabel, plt.ylabel, plt.grid, and plt.title, you may set the labels for the graph's x- and y-axes, grid, and title.
Finally, plt.show() is used to display the graph"""

# Set the colour for all groups.
diff_colors = {0 : 'brown', 1 : 'coral', 2 : 'gray'} 
def color(x):  
    return diff_colors[x]  
colors = list(map(color, k_means_algo.labels_))   

# define figure size.
plotted.figure(figsize=(10,7))
# set parameter for scatter plot.
plotted.scatter(x=X.iloc[:,0], y=X.iloc[:,2], c=colors)  
# set xlabel.
plotted.xlabel('<----- 1960 ----->',color='r', fontsize=15)
# set ylabel.  
plotted.ylabel('<----- 1962 ----->',color='r', fontsize=15) 
plotted.grid()
# set title for graph. 
plotted.title('Scatter plot for 3 Clusters', color='g', fontsize=20);  
# Getting the labels and Centroids.
getting_centroids = k_means_algo.cluster_centers_
get_label = NPP.unique(getting_predict) 
getting_centroids 

plotted.figure(figsize=(10,7))
for i in get_label:
    plotted.scatter(stand_scaled[getting_predict == i , 0] , stand_scaled[getting_predict == i , 1] , label = i)  

# define variables for graph like colour, data etc.
plotted.scatter(getting_centroids[:,0] , getting_centroids[:,2] , s = 40, color = 'b') 
# define xlabel.
plotted.xlabel('<----- 1960 ----->',color='r', fontsize=15)
# define ylabel.  
plotted.ylabel('<----- 1962 ----->',color='r', fontsize=15) 
plotted.grid() 
plotted.title('Showing Clusters with their Centroids', color='g', fontsize=20) 
plotted.legend()  
plotted.show()  

# Making the lists that will be used to get all of the cluster.
cluster_1st=[]
cluster_2nd=[] 
cluster_3rd=[] 

# Use the loop to find out what kind of info is in each cluster.
for i in range(len(getting_predict)):
    if getting_predict[i]==0:
        cluster_1st.append(urban_population_Final_Dataset.loc[i]['Country Name']) 
    elif getting_predict[i]==1:
        cluster_2nd.append(urban_population_Final_Dataset.loc[i]['Country Name'])
    else:
        cluster_3rd.append(urban_population_Final_Dataset.loc[i]['Country Name'])  
# shows the information that is in the first cluster.
cluster_1_data = NPP.array(cluster_1st)
# shows the information that is in the second cluster.
cluster_2_data = NPP.array(cluster_2nd)
# shows the information that is in the third cluster.
cluster_3_data = NPP.array(cluster_3rd)   

print(cluster_1_data)
print(cluster_2_data)
print(cluster_3_data)
first_cluster = cluster_1_data[2] 
print('Cluster_1_Country_name :', first_cluster) 

second_cluster = cluster_2_data[2] 
print('Cluster_2_Country_name :', second_cluster) 

third_cluster = cluster_3_data[3] 
print('Cluster_3_Country_name :', third_cluster) 

print('Country name :', first_cluster)
Afghanistan_country = country_name[country_name==first_cluster]
Afghanistan_country = Afghanistan_country.index.values
Afghanistan_country = urban_population_Final_Dataset_3[urban_population_Final_Dataset_3['Country Name']==int(Afghanistan_country)]  
Afghanistan_country = NPP.array(Afghanistan_country)  
Afghanistan_country = NPP.delete(Afghanistan_country, NPP.s_[:2]) 
Afghanistan_country    

print('Country name :', second_cluster) 
East_Asia_and_Pacific_country = country_name[country_name==second_cluster]
East_Asia_and_Pacific_country = East_Asia_and_Pacific_country.index.values
East_Asia_and_Pacific_country = urban_population_Final_Dataset_3[urban_population_Final_Dataset_3['Country Name']==int(East_Asia_and_Pacific_country)] 
East_Asia_and_Pacific_country = NPP.array(East_Asia_and_Pacific_country)  
East_Asia_and_Pacific_country = NPP.delete(East_Asia_and_Pacific_country,NPP.s_[:2]) 
East_Asia_and_Pacific_country  

print('Country name :', third_cluster) 
Mexico_country = country_name[country_name==third_cluster]
Mexico_country = Mexico_country.index.values
Mexico_country = urban_population_Final_Dataset_3[urban_population_Final_Dataset_3['Country Name']==int(Mexico_country)] 
Mexico_country= NPP.array(Mexico_country)  
Mexico_country = NPP.delete(Mexico_country, NPP.s_[:2]) 
Mexico_country  

#creating a list that contains years from 1960-2021
year=list(range(1960,2022))

#setting the figure size here
plotted.figure(figsize=(22,8))

#Creating first subplot that shows the population growth of Afghanistan_country.
plotted.subplot(131)
plotted.xlabel('Years')
plotted.ylabel('Population Growth') 
plotted.title('Afghanistan_country') 
plotted.plot(year,Afghanistan_country, color='#B7950B');

#Creating second subplot that shows the population growth of East_Asia_and_Pacific_country.
plotted.subplot(132)
plotted.xlabel('Years')
plotted.ylabel('Population Growth') 
plotted.title('East_Asia_and_Pacific_country') 
plotted.plot(year,East_Asia_and_Pacific_country, color='#A3E4D7');

#Creating third subplot that shows the population growth of Mexico_country.
plotted.subplot(133) 
plotted.xlabel('Years') 
plotted.ylabel('Population Growth')
plotted.title('Mexico_country') 
plotted.plot(year,Mexico_country, color='#D2B4DE');

# Curve Fitting
# selecting all columns and convert into array.
axis = NPP.array(urban_population_Final_Dataset_5.columns) 
# dropped some of the columns.
axis = NPP.delete(axis,0) 
axis = NPP.delete(axis,0) 
# change into data type as int.
axis = axis.astype(NPP.int)

# Choosing all the facts for India's urban population.
curve_fit = urban_population_Final_Dataset_5[(urban_population_Final_Dataset_5['Indicator Name']=='Urban population') & (urban_population_Final_Dataset_5['Country Name']=='India')]   

# change into array.
axes = curve_fit.to_numpy()
# dropped some of the columns.
axes = NPP.delete(axes,0) 
axes = NPP.delete(axes,0)
# change into data type as int.
axes = axes.astype(NPP.int) 

"""The linear_func function in this code specifies the linear function that will be used to fit the data, 
y = mx + c.

The curve fitting process is carried out by the create_curve_fit function using the inputs x and y. 
The given data points (x, y) are used to fit the linear function using the curve_fit function. 
The popt and pcov variables are used to calculate the fitted parameters and associated standard errors.

The stats.t.interval function and the standard errors are then used by the function to generate the confidence
interval. For both m and c, the low and high values of the confidence interval are calculated.

Finally, the function depicts the data points, the fitted function, and the confidence interval using the 
matplotlib.pyplot routines plt.plot, plt.fill_between, and others. Additionally set are the figure's dimensions, 
title, x- and y-axis labels, grid, and legend.
To utilise this code, call the create_curve_fit function with your x and y values."""

# importing some of the library.
import scipy
import numpy as NPP 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plotted
from scipy import stats 

# Set the function that needs to be fit (y = mx + c for a linear function).
def linear_func(x, m, c):
    return m*x + c

def create_curve_fit(x,y): 

    # Adjust the curves.
    popt, pcov = curve_fit(linear_func, x, y) 

    # Get the values that were fit or their standard errors
    m, c = popt
    m_err, c_err = NPP.sqrt(NPP.diag(pcov)) 

   # Figure out the bottom and top of the confidence range.
    conf_int = 0.95  # Put 95% in the confidence range
    alpha = 1.0 - conf_int 
    m_low, m_high = scipy.stats.t.interval(alpha, len(x)-2, loc=m, scale=m_err)
    c_low, c_high = scipy.stats.t.interval(alpha, len(x)-2, loc=c, scale=c_err)

    # Draw the best-fitting function or the range of confidence.
    plotted.figure(figsize=(12,6)) #set figure size.
    plotted.plot(x, y, '*', label='Data') #set data for graph.
    plotted.plot(x, linear_func(x, m, c), 'b', label='Fitted Function')
    plotted.fill_between(x, linear_func(x, m_low, c_low), linear_func(x, m_high, c_high), color='orange', alpha=0.5, label='Confidence Range') 
    plotted.title('Curve Fitting',color='g', fontsize=20) #set title for graph.
    plotted.xlabel('<----- Years ----->',color='r', fontsize=15) #set xlabel.
    plotted.ylabel('<----- Population ----->',color='r', fontsize=15) #set ylabel. 
    plotted.grid()
    plotted.legend() #set legend in graph.
    plotted.show() 
    
create_curve_fit(axis,axes) 