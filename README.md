# Spam-email-classifier

## About
This project is made as a requirement in the CSC484 course "Information retrieval".
Providing a training set of spam / ham emails, the model will learn using Multinomial Naive Bayes Model.

## Usage
The only dependancy is nltk, which is used to tokenize words in emails.
Install in debian systems using
> sudo apt install python3-nltk

Train the model using:
> nb_train(x_train, y_train)
x_train is the list of emails, y_train is the corresponding value for each email, where 0 is HAM and 1 is SPAM
The function will return the trained model.

Test the model using
> y_pred = nb_test(x_test, model, use_log = False, smoothing = False)
Make sure to load the test model prior to using the nb_test function.

You may use the provided f_score function
> f_score(y_test,y_pred)