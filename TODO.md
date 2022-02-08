## Iomed NLP Test 

During the course of this test, there were a couple more things that would've been good to be implemented. I will list them below, they are in no particular order of importance, rather things that came up during the excercise. 

1) Data Cleaning: 
	- NOW: tweets are passed through without being cleaned
	- TODO: Removal of twitter handles (@name), stopwords, and punctuations.

	In this case, the model performed fairly well considering that the text it was being fed was plagued with symbols and unimportant characters. In most cases a cleaning of the text would. Since they are tweets, a lot of information can be expressed with emojis, that's why further below I mention to use a model which has been pre-trained for the "Twittersphere", per se. 

2) Up / Down Sampling the data:
	- NOW: the training data is imbalanced (~77% to 23%), this affects the model in a negative way and can be seen in the reported metrics. 
	- TODO: use imblearn in order to upsample the minority class or downsample the majority class.

	In the EDA, it has been seen that the training data is imbalanced. This will cause the model to learn more from the majority class. I would want to consider performing different samplings (using imblearn), and test whether the effect of the class imbalance changes the performance of the model. I'd assume that it would affect since the rate of missclassification (Negative when Positive) would decrease, since it saw an equal amount of each. 

3) Test a different model that is specialized for this: 
	- NOW: Using a pre-trained model of BERT
	- TODO: use a model which was been pre-trained on twitter!
	https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

	As I mentioned above, I found this model, which has been pre-trained on twitter, which also takes into consideration the emojis. This would add more information and be of value when classifying sentiment with data that is specific to that context (in this case twitter & emojis).

4) Tuning the model
	- NOW: little tuning has been done, practically out of the box. 
	- TODO: Freezing different layers in the model and fine-tuning them has shown to be effective. For example freezing all layers, compared to all but the pool, or a combination of them. This I've seen to be effective in order to optimize the model. 
	- TODO: Consider looking at early-stopping, or decreasing the learning rate. 
