# CandyHierarchyDataCleanUp
Project for data science tool 1
# Team Members:
-  Josh Weldemariam
-  Jingxiu Hu
-  Xiao Lin
# Project Objective:
The main objective for this project is to have an accurate, consistent and complete data from the Candy Hierarchy Survey dataset. There are many issues that needs to be fixed within the dataset in order to make the dataset ready for data analysis. We plan to work on removing some errors such as typos, inconsistent capitalization and mislabeled classes. Eventhough most of the surveys were multiple choice answers allowing the participants to pick thier answer, Some of the questions including the ones asking for the participants information were open questions allowing them to answer by typing their answers which lead to more Structural errors. We will work on those columns by filtering and fixing the common errors. There are also many missing data which needs to be fixed in order to have a complete data. We will also identify feature engineering opportunity if there are any.
# Dataset attributes and metadata.
The attributes I found interesting for our dataset is age, country, state, and other comments.
A normal value example for age is 44. A noisy value example for age is "old enough".
A normal value example for country is USA. A noisy value example for country is us.
A normal value example for state is CO. A noisy value example is colorado.
For other comments, I don't think there are any normal or noisy values. Because they are just user's subjective opinions. However,
we do need to find somthing to do with these inputs.
# Tentative timeline in a tabular format

| Tables              |      Timeline            | 
|---------------------|-------------------------:|
| data collection     |   completed              | 
| data cleanup        | by the end of 5th week   |   
| transformation      | by the end of 6th week   |   
| feature engineering | by the end of 8th week   |
| statistical summary | by the end of 9th week   |  
| visualization       | by March 8th             |    

Link to Project via Binder:
https://mybinder.org/v2/gh/JingxiuHu/CandyHierarchyDataCleanUp/master?filepath=Project_Report.ipynb