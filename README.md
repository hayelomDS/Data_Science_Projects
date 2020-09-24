# **Data Science Projects**

These Repository contains various Data Science projects I worked on for acadamic and self learning purpose. The projects are presented in IPython Notebooks using Jypyter Notebook.


# Machine Learning 

# [Predicting Probability of Credit Default](https://github.com/hayelomDS/Data_Science_Projects/tree/master/credit_default)
Credit Default: Data cleaning and preparing steps followed by transforming the data and fitting 3 different models to compare. 
	
# [Sentiment Analysis](https://github.com/hayelomDS/Data_Science_Projects/tree/master/sentiment%20analysis%20for%20US%20airlines)
Sentiment Analysis on data obtained from Twitter for the major US airlines: The objective of this task is to determine if I can build a model that can accurately classify a social media tweets for the major US airlines as positive or negative based on the writers polarity. Arriving at a model that can predict and classify customers sentiment accurately can be beneficial for any sector to improve thier services and products. It can also be used as a major input to busines models and help the company achieve higher revenue. Although there are countless numbers of applications where sentiment analysis can be used, just to mention a few it can be used for decision making in the business by analyzing public opinion and developing new strategies. Sentiment analysis can also be used to gain advantage on the competitors. Other uses include improving customer service and predicting opinions on current products.

In this project I explored with different models and utilized parameter optimization techniques for each of those models to arrive at the model that has the best accuracy. The models explored with are Logistic Regression, Naïve Bayes and Support Vector Machine. The Pandas and Numpy libraries were utilized to clean and prepapre the data. The NLTK library and the Bag of Words approach were also used to transform and tokenize the words. 

## Project Implementation

The first and the most important step in any Data Science project is data cleaning and data preprocessing. In order to achieve a high accuracy, cleaning our data in a way that is required for our particular problem we are trying to solve is a vital step. Since we are dealing with a text data doing some explanatory analysis is required. After dropping features that is not relevant for this analysis the final Data Frame has 3 features which are Airline names, tweets and label. Mainly I utilized the Pandas and NumPy library’s for analysis and explanatory steps.

The Regular Expression (RE) library was utilize to help clean the text feature with the NLTK library to tokenize and stem the text. In order to arrive at high accuracy level the data we feed to the program needs to be clean. One of the approach I took was that to remove short words such words that have less than three characters. Even though the Stop words library was used to remove common words, it does not however remove every common words on every situation and removing small words was very important approach. Below is the function used to clean the data.

![](/images/clean_func.png)

## Storytelling and Visualization

To better understand the dataset we are working with, performing visualization is a necessary step in Data Science. Here I believe the use of word cloud library is the perfect fit since we are dealing with high amount of words and it shows us the most common words in a bigger font which makes it easier to visualize.

### Most frequently used words in Positive Sentiments
![](/images/Positive wordcloud.png)

### Most frequently used words in Negative Sentiments
![](/images/negative wordcloud.png)






