# [Subreddit NPL Classification Modeling](https://github.com/DSchlant/nlp_classification_reddit)

Many feel that humanity is at a major inflection point. Technological progress is rapid: cars are driving themselves and running on electricity, mundane tasks are being automated, and AI is contributing to providing solutions across many fields of study. There is much to be excited about. But - how can you be? Every week, updates on the state of the climate become more dire, the geopolitical landscape becomes more fraught, and the social fabric that binds us all together seems to fray. Where do you feel that civilization is headed? Where might your customers? Or the members of your next target market?

In an effort to utilize natural language processing (NPL) to try to determine an individual's worldview via their Reddit activity, we have used the Reddit Pushshift API to collect submissions to two Subreddit pages: r/collapse and r/futurology. The members of each community have self-identified as having conflicting views of the future of humanity. The Futurology Subreddit takes a more positive view on the path forward for humans, technology, and civilization. Members of the collapse community adopt a more pessimistic view for the decades ahead. Below are the respective descriptions for these communities, as stated by their administrators:

**Collapse of Civilization**
*r/collapse*
Discussion regarding the potential collapse of global civilization, defined as a significant decrease in human population and/or political/economic/social complexity over a considerable area, for an extended time. We seek to deepen our understanding of collapse while providing mutual support, not to document every detail of our demise.

**Future(s) Studies**
*r/Futurology*
Welcome to r/Futurology, a subreddit devoted to the field of Future(s) Studies and speculation about the development of humanity, technology, and civilization.


***Problem Statement***
The primary purpose of this analysis will be to develop a model that can accurately identify posts as originating from the Futurology Subreddit ot from Collapse, via the text content of their titles. A secondary, further application would be to apply this analysis to other social media posts or content to assess the author's general outlook on the future of civilization.

***Data Collection***
Using Puhsshift API, collected Subreddit submissions that contained self-text from Sunday April 24th back to the inception of the subreddits provided a wealth of data on both Subreddits: 28,876 posts on Futurology and 28,741 for Collapse. We proceeded with analyzing post titles alone for this study because 32% of the total submissions had self-text removed. Options for further research include utilizing the available self-text, as well as the titles for historic submissions that do not include self-text.

***Process***
Due to the wealth of available data, a holdout dataset comprising 20% of the available data was set aside to be tested after modeling process completed on training and validation sets.

Initial analysis was run on 30% of the remaining dataset, or 24% of the overall dataset. The dataset was split virtually 50%/50% between Futurology and Collapse, so this served as the initial null baseline. The model will need to demonstrate the ability to choose the correct Subreddit of origin at a higher rate than merely choosing 'Futurology' each time in order to be considered successful.

In creating a custom stoplist for the model, an effort was made to add forms of future/collapse, as well as the subreddit's name. This negatively affected performance. However, it may provide what may be a sounder methodology for identifying an individual's worldview or expectations for the future via their post language.

***Modeling***
Many supervised learning models were tested in order to identify the model best fit for classifying the submissions, after either count vectorizing the data or vectorizing the data via TF-IDF. The models run included:

* Multinomial Naive Bayes
* Gaussian Naive Bayes
* Logistic Regression
* K Nearest Neighbors Regression
* Decision Tree 
* Random Forest
* AdaBoost
* Ensemble Analysis

The best performing models incorporated Naive Bayes, with the model utilizing TF-IDF vectorization being the most accurate. Further reading regarding TF-IDF vectorization provided by experienced data scientists can be found below. TFI-DF does not necessarily rely on word count alone. The method provides more weighting to a word if it occurs often in a certain submission title, but will reduce its influence in the model if that same word shows up in many of the titles.

Naive Bayes Classifier is a supervised learning method that will generate the probability that, in this context, a post belongs to a certain thread given the words that are in the title, and selects the thread with the highest probability. This is a simple explanation, but further reading can be found below as well.

***Results***
This model, after incorporating the customized list of stop-words, generated an accuracy rate of the test set (using 30% after the holdout was removed) of 80.8%. Using just the list of standard English stop-words yielded an accuracy rate of 82.3% on this test set.

When this model was fit run on 100% of the training/validation dataset, the model had an accuracy rate of 82.8%, improving as the model saw more data. The model performed with **81.67% accuracy on the holdout set**, which is 20% of the overall dataset initially scraped from Reddit. This is well above the 50.3% baseline derived from the percentage of the majority class in this dataset.

***Qualitative Analysis***
Qualitative analysis of the words and terms with the most impact on the model were telling:
* Collapse Subreddit
    * Community-oriented, supportive
    * Strong survivalist community
    * Humorous, profane
    * Climate Change obsessed

* Futurology Subreddit 
    * Extending life, longevity
    * Automation/robotics 
    * Luxury, leisure oriented


***Conclusion***
We have developed a model that can predict whether Reddit user's post was intended for a techno-positive, future-optimistic forum or a thread with a more fatalistic view with 82% accuracy when introduced to new data. As the model encounters more data it improves in its accuracy. Using the findings of this NLP model may provide companies insight into a user's worldview or expectations of the future based on text that they have produced.

***For Further Research***
*Self-Text Analysis*
Quick analysis was performed using the same Naive Bayes model on both the title and self-text data. Titles alone were used for the previous analysis since self-text had been removed for such a large percentage of the posts. When the available self-text was added to the model and some hyperparameters tuned, the model achieved 84.64% accuracy on the validation set.

*Sentiment Analysis*
Develop and add sentiment analysis to model so model can better asses author's tone or intent. May lead to more accurate modeling, especially when the classes to be identified contain similar text, as was the case in this study.

*Expand Research to other Subreddits*
This study partially intended to develop an understanding of the language used by Reddit users who identify themselves as having a certain worldview. Incorporating additional subreddits with similar attitudes could contribute to the development of a model that can predict a user's worldview based on their activity on other platforms or forums.  

***Further Reading***
* https://towardsdatascience.com/text-vectorization-term-frequency-inverse-document-frequency-tfidf-5a3f9604da6d
* https://www.linkedin.com/pulse/count-vectorizers-vs-tfidf-natural-language-processing-sheel-saket/
* https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html



# [Housing Price Modeling](https://github.com/DSchlant/ames_housing)

### Data Science Problem:

I am affiliated with the Home Valuation Consultative Services, Office for Faculty and Staff Housing Support at Iowa State University. 

It is a small data consultancy housed within the Iowa State University Office of the Provost, which provides free consulatative services to ISU employees. 

Faculty come here thinking they will stay in beautiful Ames Iowa forever after landing their dream job at ISU. They buy a home and settle in. They may start making renovations to their house, after all, this is where they will be hosting their retirement party! 

However, a few years down the line, they get a call from The University of Nebraska, Minnesota, or God forbid, Iowa. They are made an offer they cannot refuse, and one thing leads to another, and they are headed to greener pastures come next fall. This leaves them with less than a year to get their house in condition to be sold, be it to the new crop of ISU professors looking for the house they will retire in, a family interested in moving out of the hustle and bustle of Des Moines, or otherwise.

Part of what our team does is consult with professors who are interested to learn what they can do in order to try to maximize the price they can ask for their home. We do this free of charge, as part of what we offer faculty to entice them to join Iowa State in the first place.

Our team utilizes past transactions in Ames, from 2006-2010, in order to run regression analyses to develop the recommendations that we pass on to the individuals who utilize our services. There are 79 house characteristics available to analyze.

The process that our team endeavors to take on is to create a model that incorporates features of a home that cannot be changed, such as neighborhood, lot size, age so that we can accurately predict home prices, and also include features that can be relatively easily changed in the few short months that employees have before they must sell the house. This allows us to build a strong model that can provide clear recommendations to our clients.


### Data Cleaning and EDA

To arrive at a clean dataset, you will find in the Data Cleaning notebook our process of dropping extraneous columns, consolidating some columns, and imputing missing data. We add the numberic values for our ordinal feautures as well, incorporating the ranking systems provided in the data dictionary. Several outliers in our training set are removed.

We split our training dataset into training and validation sets, and proceed to analyze our training set further.

We studied the correaltion between salesprice and the numeric features available in order to get a sense for the relationship between these features and price, with an eye for what to include in the final regression analysis that we perform.

We also perform edsome additional analyses on categorical variables to provide further clarity.


### Preprocessing and Modeling

Using a bit of a trial and error process, with an eye for our overall data science problem, we selected 27 features to analyze in our regression models that produced low error and served to contribute to recommendations that we can offer to the college staff. Utilizing pipeline transformations, the selected data was scaled and one-hot-encoded. The target data was also log transformed so it could be normalized, which produced better fitting models.

We proceeded to run a series of permutations of regression analyses in order to find the regression model that best limits the mean squared error. 

The model that ultimately was selected was the Ridge CV regression usin:
* the 27 selected features
* log transformation of target
* an alpha of 0.2

This model produced the below R-Square Values:

* Train: 0.9334
* Validation: 0.9147
* K-Folds (5 Folds): 0.9187

The Mean Sqaured Error for this regression was 376,816,821. This is much better than the mean-squared-error of 4,417,414,053 produced by our Baseline Model, and better than the Linear and Lasso Regression models that we calcualted.


### Recommendations for Homeowners:

We can make the below advisements to homeowners who consult with us (keeping in mind that these values re to be interpreted as the change if all other variables were held equal).

* Installing Central Air can add ~6% to home price.
* Pavig driveway can add ~5.5% to home price.
* Fences (of any kind) slghtly hurt value of home.
* Having a garage that is in poor quality can decrease a home price by 18% versus not having a garage. Conversely, a garage of good quality can add ~8% to a home value.
* Improving kitchen quality (surfaces, appliances) from poor to good can increase house price by 6%, from poor to excellent can improve a house price by 12%.
* Improving heating quality from poor to fair can increase home price 44%.
* Improving overall condition of house from poor to average can increase your house price by ~20%.
* If 1 bathroom in above ground living area, adding a second full bathroom only increase house price by ~5%, may not be worth cost and hassle.
* The exterior of the house is very important. If your house has asbestos shingles or comon brick as one of its primary exterior materials, considering replacing it. Asphalt shingles, brickface, stucco, and cinderblock is best.



