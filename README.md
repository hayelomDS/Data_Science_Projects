# **Data Science Projects**

These Repository contains various Data Science projects I worked on for acadamic and self learning purpose. The projects are presented in IPython Notebooks using Jypyter Notebook.

# [Predicting Probability of Credit Default](https://github.com/hayelomDS/Data_Science_Projects/tree/master/credit_default)
I will be working on a dataset found on the UCI Machine Learning Repository. The dataset has information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. I choose to analyze this dataset because of my curiosity in finding what signals could indicate that can help predict borrower defaulting at least a month in advance. If that can be predicted accurately, lenders could decrease the avaliable limit or possibly close the account before it's used more.

Data collection was done through the website of UCI Machine Learning Repository. The dataset has lots of irrelevant features I was able to drop. All though this dataset did not have missing values, it did have many values that was inconsistent. After doing some analysis by comparing important features against each other using comparasion operators, I was able to filter out the inconsistent data. 

After the initial cleaning steps I utilized the KBinsDiscretizer function to put the age groups into bins and encoded the nominal and ordinal features. I used the min max scaler to scale some features as well before I started with my model comparison. 

### Plotting Age Distribution 
![](/images/age_dist.png)

I compared the performance of Logisitc Regression, Decision Tree and Random forest models. The Descion Tree model performed better based on accuracy using only depth of 3. Allthough they were all around 80% accuarte, I found that Decision tree performed better than the Random forest as well. I utilized hyperparameter tuning which helped with accuracy. With the default depth, My accuracy was only 72% however after changing the depth to 3-5, the accuracy went up to 82% for all models. 
	
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
![](/images/Positive_wordcloud.png)

### Most frequently used words in Negative Sentiments
![](/images/negative_wordcloud.png)

## Model Testing

In conclusion although I believe SVM gave us the best accuracy, it is not an ideal approach to take in the real world because of the computational cost. Logistic Regression gave us a very similar result to SVM and it’s a better model to use. The Naïve Bayes model was lower in accuracy compared to the other models however I still prefer to use the Naïve Bayes model due its simplicity to use and its flexibility to handle high dimensional feature. NLP is still growing as a field and there are still lots of work and research to be done in this field. I was able to obtain accuracy in the high 70’s with all class labels and over 90% for 2 class labels on every model I trained. However I believe there are more I can do and implement particularly in the data cleaning and preprocessing steps to get a higher results.
