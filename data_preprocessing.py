import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle

df=pd.read_csv("toxic_comments_preprocessed.csv")

df.drop(["id","identity_hate","insult","obscene","set","severe_toxic","threat"],axis=1,inplace=True)
df['toxicity'].value_counts()
toxic=df['toxic'].value_counts()

ps=EnglishStemmer()
# analyzer=TfidfVectorizer().build_analyzer()
# def steemer(comment):
#     return (ps.stem(w) for w in analyzer(comment))


vectorizer=TfidfVectorizer(stop_words=stopwords.words("english")+list(string.punctuation),lowercase=True,max_features=1500,min_df=5,max_df=0.7)
analyzer=vectorizer.build_analyzer()
df['comment_text'].apply(lambda x:(ps.stem(w) for w in analyzer(x)) )
comment_bow=vectorizer.fit_transform(df['comment_text']).toarray()
x_train,x_test,y_train,y_test=train_test_split(comment_bow,df['toxic'].values,test_size=0.25,random_state=0)
classifier=MultinomialNB()
classifier.fit(x_train,y_train)
pred=classifier.predict(x_test)
cm = confusion_matrix(y_test, pred)
acc_s = accuracy_score(y_test,pred)


pickle.dump(classifier,open('comment_classifer','wb'))
pickle.dump(vectorizer,open('comment_vectorizer','wb'))