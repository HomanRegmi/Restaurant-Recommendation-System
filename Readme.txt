Background:

This project was completed in the partial fullfilment of the Data Science Certification Course conducted by The Data Incubator (TDI). In the project, I utilized two sets of json datasets from YELP: yelp_academic_business.json and yelp_academic_review.json where I extracted data on restaurants and built a restaurant recommendatation system for the customers looking for a restaurant of his/her choice in the relevant cities- Toronto, Las Vegas and Phoenix. You have option to add more cities in the code to have restaurant recommendation in the respective cities.


Recommendation System:

The machine learning algoritm I used to develop this recommendation system is K-Nearest Neighbors (KNN) which is one of the most used supervised machine learning algorithm in data science field. I used this algorithm to identify major categories of the restaurant business. 

TF-IDF was used to give the weight of the informative words. Average stars were calculataed for each business_id.

I picked three cities of North American continent- Toronto, Las Vegas and Phoenix as these were the top three cities with highest restaurant numbers in the dataset.


In my recommendation system, beside choosing city, food categories and average stars, you can also match the review with your own preferred review.

Steps to follow:

1. The necessary JSON files are located in the 'raw data' folder.

2. Run the 'Restaurant_recommendations with EDA.ipynb' script in the codes folder for EDA and observe 			how the models function. This will also store the processed data to the 'processed data' folder.

3. The top three cities were picked. If you want to add a city, enter its name in 'Restaurant_recommendations using EDA.ipynb' and execute the entire code. As a result, the processed data now includes the city you specified.

4. The dashboard code is the 'Restaurant_recommendation_system.py' file in the codes folder, where you can make all of your unique customizations.

5. To use the dashboard, follow these steps: Navigate to the directory and type the command "streamlti run Restaurant_recommendation_system.py" in the terminal.
