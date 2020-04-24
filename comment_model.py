import pickle
# from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# from nltk.stem.snowball import EnglishStemmer
# from nltk.corpus import stopwords
# ps=EnglishStemmer()
# analyzer=TfidfVectorizer().build_analyzer()
# def steemer(comment):
#     return (ps.stem(w) for w in analyzer(comment))
model=pickle.load(open('comment_classifer','rb'))
vectorizer=pickle.load(open('comment_vectorizer','rb'))
a=input("enter the message : ")
l=[a]
print(model.predict(vectorizer.transform(l)))   