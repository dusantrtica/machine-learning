from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=10)

# Count the word occurences
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)

# we transform word occurences into tf-idf - that is more meaningful
# TfidVectorizer = CountVectorizer + TfIdfTransformer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

model = MultinomialNB().fit(x_train_tfidf, training_data.target)

new = ['This has nothinng to do with church or religion', 'Software engineering is getting hotter and hotter nowdays']
new_counts = count_vector.transform(new)
x_new_tfidf = tfidf_transformer.transform(new_counts)

predicted = model.predict(x_new_tfidf)
for doc, category in zip(new, predicted):
    print('%r -------> %s' % (doc, training_data.target_names[category]))