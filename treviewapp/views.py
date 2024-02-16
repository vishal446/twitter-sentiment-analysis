from django.shortcuts import render
import joblib
from django.http import HttpResponse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer




def predict_sentiment(text):
    # Load the pre-trained model
    model = joblib.load('treviewapp/trained_model.sav')
    # Assuming you have a trained vectorizer
    print("passed model")
    vectorizer = joblib.load('treviewapp/senan_vectorizer.pkl')
    # Transform text data to numeric vectors
    numeric_input = vectorizer.transform([text]).toarray()
    # Perform prediction (adjust this part based on your model and vectorization process)
    prediction = model.predict(numeric_input)

    return prediction[0]

def home(request):
    if request.method == 'POST':
        user_input = request.POST['user_input']
        prediction = predict_sentiment(user_input)
        if prediction==1:
            prediction='Positive!'
        else:
            prediction='Negative!'
        return render(request, 'add_tweet.html', {'user_input': user_input, 'prediction': prediction})
    else:
        return render(request, 'add_tweet.html')
