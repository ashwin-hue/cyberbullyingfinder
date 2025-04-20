import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import time 

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict_emotion(comment):
    comment_tfidf = vectorizer.transform([comment])
    prediction = model.predict(comment_tfidf)[0]
    label_mapping_inverse = {0: 'normal👍', 1: 'offensive 🤢🤮', 2: 'hatespeech😠'}
    return label_mapping_inverse[prediction]


while True:
    user_input = input("Enter a comment (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    emotion = predict_emotion(user_input)
    print(f'The detected emotion is: {emotion}')
    #if emotion == "offensive 🤢🤮":
  #      print("you got a time out of 60 sec")
   #     time.sleep(5)
    #else:
     #   continue