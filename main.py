import glob
import os
import nltk
import math

def load_data(directory):
    x = []
    y = []
    for f in glob.glob(os.path.join(directory,"HAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(0)
    for f in glob.glob(os.path.join(directory,"SPAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(1)
    return x,y

def nb_train(x, y):
    nbHam = 0 # count for ham emails
    nbSpam = 0 # count for spam emails
    
    ham_fd = dict() # ham word frequency tracker
    spam_fd = dict() # spam word frequency tracker

    vocab = set() # keep track of all unique words
    
    # Loop each email in our corpus
    for email, isSpam in zip(x, y):
        # Tokenization
        words = nltk.word_tokenize(email)

        vocab.update(words)

        if isSpam == 0:  # HAM email
            nbHam += 1
            # Loop each word (token) in current email
            for word in words:
                if word in ham_fd:
                    ham_fd[word] += 1
                else:
                    ham_fd[word] = 1
        else: # SPAM email
            nbSpam += 1
            for word in words:
                if word in spam_fd:
                    spam_fd[word] += 1
                else:
                    spam_fd[word] = 1
    
    V = len(vocab) # |V|, will be used for add-1 smoothing to prevent zero probabilities
    model = {
        'ham_count': nbHam,
        'spam_count': nbSpam,
        'ham_fd': ham_fd,
        'spam_fd': spam_fd,
        'vocab_len': V
    }
    
    return model

def nb_test(docs, trained_model, use_log = False, smoothing = False):
    ham_count = trained_model['ham_count']
    spam_count = trained_model['spam_count']
    ham_fd = trained_model['ham_fd']
    spam_fd = trained_model['spam_fd']
    totSpamWords = 0
    totHamWords = 0

    # Some value preparations

    # Count total spam words and ham words
    for i in spam_fd.values():
        totSpamWords += i
    for i in ham_fd.values():
        totHamWords += i

    # Iterating the test emails
    predictions = []
    for doc in docs:
        words = nltk.word_tokenize(doc)

        # Count probability of spam emails and ham emails in prior
        if use_log:
            spamEmailsProb_log = math.log(spam_count / (spam_count + ham_count))
            hamEmailsProb_log = math.log(ham_count / (spam_count + ham_count))
        else:
            spamEmailsProb = spam_count / (spam_count + ham_count)
            hamEmailsProb = ham_count / (spam_count + ham_count)

        for word in words:
            if smoothing:
                spamWordProb = (spam_fd.get(word, 0) + 1) / (totSpamWords + trained_model['vocab_len'])
                hamWordProb = (ham_fd.get(word, 0) + 1) / (totHamWords + trained_model['vocab_len'])
            else:
                spamWordProb = spam_fd.get(word, 0) / totSpamWords
                hamWordProb = ham_fd.get(word, 0) / totHamWords
            
            if use_log:
                spamEmailsProb_log += math.log(spamWordProb)
                hamEmailsProb_log += math.log(hamWordProb)
            else:
                spamEmailsProb *= spamWordProb
                hamEmailsProb *= hamWordProb

        if use_log:
            if(hamEmailsProb_log > spamEmailsProb_log):
                predictions.append(0)
            else:
                predictions.append(1)
        else:
            if(hamEmailsProb > spamEmailsProb):
                predictions.append(0)
            else:
                predictions.append(1)
    return predictions

def getPrecisionRecall(y_true, y_pred):
    TP = 0 # true pos
    TN = 0 # true neg
    FP = 0 # false pos
    FN = 0 # false neg

    precision = 0
    recall = 0

    F1 = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 1 and pred == 0:
            FN += 1
        else:
            FP += 1

    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    return precision, recall



def f_score(y_true, y_pred):
    precision, recall = getPrecisionRecall(y_true, y_pred)

    if (precision + recall) > 0:
        F1 = (2 * precision * recall) / (precision + recall)
    
    return F1

x_train, y_train = load_data("./SPAM_training_set/") # LOAD the training data set
model = nb_train(x_train, y_train) # train the model
x_test, y_test = load_data("./SPAM_test_set/") # LOAD the test data set
y_pred = nb_test(x_test, model, use_log = True, smoothing = True) # get the predicted results using the trained model
print(f_score(y_test,y_pred))