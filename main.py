from optparse import Option
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

st.set_page_config(
    page_title='Bengaluru House Price Prediction'
)

with st.sidebar:
    selected = option_menu(
        menu_title='Main Menu',
        options = ['About', 'EDA', 'Data preprocessing', 'Model', 'Contact'],
        icons= ['file-earmark-text-fill','pie-chart-fill','reception-4', 'graph-up', 'envelope-fill'],
        menu_icon='list-ul',
        default_index=0

    )

df = pd.read_csv('Bengaluru_House_Data.csv')

data =df.copy()
data.drop(columns=['area_type','availability','society'],inplace=True)
## Handeling the missing values
data=data.dropna()
###  change the size from object to interger 
data['BHK'] = data['size'].str.split().str.get(0).astype(int)

# we will take the average of the range
def convertRange(x):
    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convertRange)
data = data.dropna(subset=['total_sqft'])
### make price per square feet to detict outliears
data['price_per_sqft'] = data['price'] *100000 / data['total_sqft']
#Change the locations that happen less than 10 to others to decrease the amount of unnessary locations
d2 = data.copy()# for data preprocessing 

data['location'] = data['location'].apply(lambda x:x.strip())
location_count = data['location'].value_counts()
location_count_less_10 = location_count[location_count<=10]
data['location'] = data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)  
# Outlier detection and removal
data = data[((data['total_sqft']) >= 300)]

def remove_outliers_sqft (df):
    
    #Create an empty dataframe df_output to store the filtered data.
    
    df_output = pd.DataFrame()
    
    #Group the input dataframe df by the location column using the groupby method.
    for key,subdf in df.groupby('location'):
        
        #For each group, calculate the mean m and standard deviation st of the
        #price_per_sqft column using NumPy's mean and std functions.
        
        m = np.mean(subdf.price_per_sqft)
        st= np.std(subdf.price_per_sqft)
        
        #Filter the group to only include rows where the price_per_sqft 
        #value is within one standard deviation of the mean using boolean indexing.
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        
        #Concatenate the filtered groups together into the df_output dataframe 
        #using pandas' concat method.
        df_output = pd.concat([df_output, gen_df],ignore_index =True)
        
    #Return the filtered df_output dataframe.
    return df_output

data=remove_outliers_sqft(data)
d3 = data.copy()# for data preprocessing 

def bhk_outlier_remover(df):


    # Create an empty numpy array exclude_indices to store the indices of the rows 
    #to be excluded from the filtered data.
    exclude_indices = np.array([])
    
    # Group the input dataframe df by the location column using the groupby method. 
    #This creates separate sub-dataframes for each unique location.
    
    for location, location_df in df.groupby("location"): 
        bhk_stats = {}
        # here we make a dict that contain the mean of BHK expamle 1, it's std and the group
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean (bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
                }
        for bhk, bhk_df in location_df.groupby('BHK'): 
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df [bhk_df.price_per_sqft<(stats['mean'])].index.values)
                #Drop the rows with the indices in the exclude_indices array from the input dataframe df using pandas' drop method
    return df.drop(exclude_indices, axis='index')

data = bhk_outlier_remover(data)
## Clean data
data.drop(columns=['size','price_per_sqft'],inplace=True)

# Make dummies one hot encoding
final_DF = data.copy()
dummies = pd.get_dummies(final_DF.location)

#concatnate the data
final_DF = pd.concat([final_DF, dummies], axis = 1)
final_DF = final_DF.drop(['location'], axis = 1)

X = final_DF.drop(['price'], axis = 1)
Y = final_DF['price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso,Ridge

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

from sklearn import metrics
mape = metrics.mean_absolute_percentage_error(y_test,predictions)

if selected == 'About':

    st.title('Bengaluru House price data')
    
    image = Image.open('wp7537471.jpg')
    st.image(image, caption='Bengaluru', width=500, use_column_width=True)

    more_ifo=st.checkbox('More inforamtion about Bengaluru city')
    if more_ifo:
        st.markdown(''' Bengaluru, also known as Bangalore, is the capital city of the Indian state of Karnataka. It is located in the southern part of India and is known as the "Silicon Valley" of India due to its thriving information technology industry.''')

    st.header('The Context of the data')
    st.markdown("Buying a home, especially in a city like Bengaluru, is a tricky choice. While the major factors are usually the same for all metros, there are others to be considered for the Silicon Valley of India. With its help millennial crowd, vibrant culture, great climate and a slew of job opportunities, it is difficult to ascertain the price of a house in Bengaluru.")
    df = pd.read_csv('Bengaluru_House_Data.csv')
    st.subheader('Sample from the dataset')
    btn = st.button('Sample')
    if btn:
        st.write(df.sample(6))

    st.header('The Size of the dataset')
    st.write(f"The number of rows  {df.shape[0]}")
    st.write(f"The number of columns {df.shape[1]}")

    st.header('The Goal 🎯')
    st.markdown("The goal is to develop an exploratory data analysis (EDA) and a predictive model that can assist users in determining the price of a house. By analyzing the relevant features and variables, the EDA can provide insights into the factors that affect house prices in a particular area. The predictive model, on the other hand, can use these insights to estimate the price of a house based on its features and characteristics. The ultimate aim is to provide users with a tool that can help them make informed decisions when buying or selling a house.")

if selected == 'EDA':
    st.title('Exploratory data analysis (EDA)')

    st.subheader('The Distribution of the dataset')

    fig = make_subplots(rows=3, cols=2)

    price = go.Histogram(x=df['price'], nbinsx = 30,name='Price')
    balcony = go.Histogram(x=df['balcony'], nbinsx=30, name='Balcony')
    size = go.Histogram(x=df['size'],nbinsx=30, name='Size')
    total_sqft = go.Histogram(x=df['total_sqft'], nbinsx=30,name='Total sqft')
    bath = go.Histogram(x=df['bath'], nbinsx=30, name='Bath')
    availability = go.Histogram(x=df['availability'], nbinsx=30, name='Availability')

    fig.append_trace(price, 1, 1)
    fig.append_trace(balcony, 2, 2)
    fig.append_trace(size, 3, 1)
    fig.append_trace(total_sqft, 1, 2)
    fig.append_trace(bath, 2, 1)
    fig.append_trace(availability, 3, 2)
    fig.update_layout(title='Histograms of Property Features', height=800, width=800)
    st.plotly_chart(fig)

    st.markdown('There are right skewed in the price and total sqft that means there is many outliers and normally there is 2 BHK, 2 bathroom, balconies and received thier property at 19 December')

    btn_check= st.checkbox('Show histogram for each column')
    if btn_check:
        st.subheader('Histogram')
        var = st.selectbox('Select Variable',df.columns)
        fig = px.histogram(df, x=var, nbins=20)
        st.plotly_chart(fig)

    st.subheader('Statistic of the data')

    numerical_stat=st.checkbox('The Statistic of numerical data')
    if numerical_stat:
        st.write(df.describe())
    categorical_stat=st.checkbox('The Statistic of categorical data')
    if categorical_stat:
        st.write(df.describe(include='object'))

    st.subheader("What is the average price of the properties in the dataset?")

    box_var =st.selectbox('Box Plot',df.columns, index=8)
    fig = px.box(df, y=box_var)
    st.plotly_chart(fig)
    st.markdown(f'### Statsitic of {box_var} ')
    st.write(df[box_var].describe())
    if box_var == 'price':
        st.markdown('The average price is 112.5 lakh but there is alots of outliers and it is make the mean right skewed so it will be better to depend on the median price instead 72 lakh')

    st.subheader('How does the number of bathrooms affect the price of a property ?')
    st.markdown('### Bar plot')

    colx =st.selectbox('Select X-axis',df.columns, index=6)
    coly =st.selectbox('Select Y-axis',df.columns, index=8)

    fig = px.bar(df, x=colx, y=coly)
    st.plotly_chart(fig)
    st.markdown('The number of bathrooms are effect the price but also it depends on the loacton because the price varies in same number of bathrooms ')

    st.subheader('What is the most area type used ?')
    fig = px.pie(df, values='price', names='area_type')
    st.plotly_chart(fig)
    st.markdown('The is the most area type used is Super built-up Area by 54.5% and Super built-up area is a term commonly used in real estate in India to describe the total area of a property, including both the carpet area and the common area. The carpet area refers to the actual physical area of the apartment or house, whereas the common area includes spaces such as the lobby, staircase, elevator, and other shared amenities.	')


    st.subheader('What is the relation total squard area of the property with price  ?')
    st.subheader('Line Chart ')
    st.line_chart(data, x='price', y ='total_sqft')
    st.markdown('There are a positive relation in general between the total squard area of the property with price but alos the location have an effect ')

if selected == 'Data preprocessing':

    

    d1 =df.copy()

    st.title('Data Preprocessing')
    st.subheader('Handeling the missing values')

    btn = st.button('Fix the missing values')
    if btn:
        d1.drop(columns=['society'],inplace=True)
        d1=d1.dropna()
        fig = plt.figure(figsize=(15,8))
        sns.heatmap(d1.isnull())
        st.pyplot(fig)
    else:
        fig = plt.figure(figsize=(15,8))
        sns.heatmap(df.isnull())
        st.pyplot(fig)

    st.markdown('We have four columns have missing values to solve this problem we remove society column because it have more than 40% missing and drop the none value form the other column because they have very small amount less than 5% ')
    st.subheader('Outlier detection')
    btn_ol = st.radio('Choose ',('After',"Before"))

    if btn_ol == 'After':
        fig = px.histogram(d3, x ="price_per_sqft")
        st.plotly_chart(fig)
    else:
        fig = px.histogram(d2, x="price_per_sqft")
        st.plotly_chart(fig)
    st.markdown('We remove the outlier of the price per sqauer feet by keep only one stander deviation')

    st.subheader('The Correlation')
    correlation= data.corr()
    fig=plt.figure(figsize=(20,7))
    sns.heatmap(correlation,annot=True)
    st.pyplot(fig)
    st.markdown('There is stronge positive relation between totale square feet and price by 80% and there is stronge positive relation between BHK and bathroom by 87%')

if selected == 'Model':
    st.title('Regression model')
    fig = plt.figure(figsize=(15,8))
    plt.scatter(y_test, predictions, c='red', label='Actual')
    plt.scatter(y_test, y_test, c='blue', label='Predicted')
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Predicted versus actual values")
    st.pyplot(fig)

    evaluation = st.button('Model Evaluation')

    if evaluation:
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Score", f'{round(model.score(x_test, y_test)*100,2)}%')
        col2.metric("R2",  f"{round(metrics.r2_score(predictions, y_test)*100,2)}%")
        col3.metric("MAPE",f'{round(mape,2)}%')

    def model_predict_function(location, total_sqft, bath,balcony,bhk):
        index = np.where(X.columns == location)[0][0]
        
        x = np.zeros(len(X.columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = balcony
        x[3] = bhk
        if index >= 0:
            x[index] = 1
        
        return model.predict([x])

    st.subheader('The Conclusion')
    btn_cl = st.checkbox('Show')
    if btn_cl:
        st.markdown('''
Total square feet: The larger the total square feet of a property, the higher its price is likely to be. This is likely because larger properties generally offer more living space and amenities, and are therefore more valuable.

BHK: The number of bedrooms, or BHK, is an important factor in determining the price of a property. Generally, properties with more bedrooms are more expensive, as they offer more living space and can accommodate larger families.

Date of purchase: The fact that the apartment is likely to be purchased on December 19th is not likely to have a direct impact on the price of the property. However, it is possible that the timing of the purchase could affect negotiations or availability of properties on the market.

Property specifications: The average property in the area appears to have 2 bedrooms, 2 bathrooms, and 2 balconies, and measures the Super built-up area. These specifications can also affect the price of a property, as they can impact the overall living experience and amenities offered by the property.''')
    st.subheader('Sample from the dataset')
    btn = st.button('Sample')
    if btn:
        st.write(data.sample(4))

    st.subheader('Predict the house price')
    location =st.selectbox('Enter your Location', data['location'].unique())
    total_sqft =st.text_input('Enter the number of total square feet',1000)
    bath = st.text_input('Enter the number of Bathrooms',1)
    balcony = st.text_input('Enter the number of balconies',1)
    bhk =st.text_input('Enter the number of BHK',1)
    btn_res = st.button('Submit')

    result=model_predict_function(location, total_sqft, bath, balcony,bhk)

    if btn_res:
        st.markdown("#### Predicted The House Price ")
        num = result[0]
        st.write(f" The price of the house is equal :",format(round(num,2),',.2f'),"lakh" )
        st.success('Success')

    dollar_btn = st.button('Convert to dollar')
    if dollar_btn:
        num = result[0]
            # Convert 1 lakh to USD
        usd_value = num * 100000 / 74.5
        st.write(f"{num:.2f} lakh INR is equal to {format(round(usd_value,2),',.2f')} USD")

if selected == 'Contact':
    st.title('Contact 📞')
    
    image = Image.open('contact-us-1908762_1280.png')

    st.image(image, caption='', width=500, use_column_width=True)

    st.markdown('''Thank you for visiting my contact page. If you have any questions or would like to connect, please don't hesitate to reach out to me. As a data scientist, I am passionate about using data-driven approaches to solve complex problems and make informed decisions. I am always interested in collaborating with others who share this passion.''')

    st.markdown("[Send me an Gmail](mailto:islamatef55035@gmail.com)   My Gmail 📧")
    st.markdown("[See others projects in kaggle](https://www.kaggle.com/islamatef4)   My Kaggle account 📄")
    st.markdown("[Contact me in Linkedin](https://www.linkedin.com/in/islam-atef-69622b214)   My Linkedin account 📲 ")

