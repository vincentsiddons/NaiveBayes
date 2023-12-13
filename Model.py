import functools
import operator
import math
import os

#Note that this model assumes classifications are single character strings
class Model:
    #attributes
    model_type = ''
    training_csv = ''
    test_csv = ''
    dev_num_high = 0
    dev_num_low = 0

    curr_directory = os.path.dirname(os.path.abspath(__file__))

    #argument constructor
    def __init__(self,model="standard", train=os.path.join(curr_directory, "training.txt"),
                 test=os.path.join(curr_directory, "test.txt"), num1=1, num2=2):
        self.model_type = model
        self.training_csv = train
        self.test_csv = test
        self.dev_num_low = num1
        self.dev_num_high = num2

    #no argument constructor
    @classmethod
    def no_argument_init(cls):
        return cls()
    
    #toString method, called when object is printed
    def __str__(self):
        return "You have the model: " + self.model_type

    @staticmethod
    def check_arr(arr, value):
        for i in range(0, len(arr)):
            if arr[i] == value:
                return True
        return False
    
    #only works for binary classifiers
    def calculate_precision(self, sentences, classes, keys):
        tp = 0
        fp = 0
        for i in range(0, len(sentences)):
            sentences[i] = sentences[i][len(sentences[i]) - 1:len(sentences[i])]
            if classes[i] == sentences[i][0] and sentences[i][0] == keys[0]:
                tp += 1
            elif sentences[i][0] == keys[1] and classes[i] == keys[0]:
                fp += 1
        try:
            precision = tp/(tp + fp)
            return precision
        except:
            return "undf"
        
    #only works for binary classifiers
    def calculate_recall(self, sentences, classes, keys):
        tp = 0
        fn = 0
        for i in range(0, len(sentences)):
            sentences[i] = sentences[i][len(sentences[i]) - 1:len(sentences[i])]
            if classes[i] == sentences[i][0] and sentences[i][0] == keys[0]:
                tp += 1
            elif sentences[i][0] == keys[0] and classes[i] == keys[1]:
                fn += 1
        try:
            recall = tp/(tp + fn)
            return recall
        except:
            return "undf"
        
    #Creates dictionary of keys that are class indicators and values that count all instances of the class
    def populate_class_dict(self, arr, class_dict):
        for i in range(0, len(arr)):
            try:
                class_dict[arr[i]] += 1
            except:
                class_dict[arr[i]] = 1
        return class_dict
    #Creates dictionary of keys that are arrays and values that are the sum of their len
    def populate_sum_dict(self, arr, sum_dict):
        for i in range(0, len(arr)):
            if i not in range(self.dev_num_low, self.dev_num_high):
                try:
                    sum_dict[arr[i][len(arr[i]) - 1]] += len(arr[i]) - 1
                except:
                    sum_dict[arr[i][len(arr[i]) - 1]] = len(arr[i]) - 1
        return sum_dict
    #Counts each instance of a word in its respective class
    def populate_word_dict(self, arr, word_dict):
        for j in range(0, len(arr)):
            for i in range(0, len(arr[j]) - 1):
                if j not in range(self.dev_num_low, self.dev_num_high):
                    try:
                        word_dict[arr[j][i] + arr[j][len(arr[j]) - 1]] += 1
                    except:
                        word_dict[arr[j][i] + arr[j][len(arr[j]) - 1]] = 1
        return word_dict
    #Calculates the size of the vocabulary
    def calculate_vocab(self, arr):
        tracking_arr = []
        for j in range(0, len(arr)):
            for i in range(0, len(arr[j]) - 1):
                if j not in range(self.dev_num_low, self.dev_num_high):
                    if Model.check_arr(tracking_arr, (arr[j][i])) == True:
                            i += 1
                    else:
                        tracking_arr.append(arr[j][i])
        return tracking_arr
    #Calculates prior probs for each class and cleans sentence array, for the dev set
    def calculate_priors(self, arr, dict):
        priors = []
        dict_keys = list(dict.keys())
        sentences = arr[self.dev_num_low:self.dev_num_high]
        for i in range(0, len(sentences)):
            sentences[i] = sentences[i][3:len(sentences[i]) - 1]
            for i in range(0, len(dict_keys) - 1):
                priors.append(math.log10(dict[dict_keys[i]]))
        for i in range(0, len(sentences)):
            sentences[i] = sentences[i].split(',')
        return priors, sentences
    
    #calculates the log probability for each class for each document
    def calculate_probs(self, prior, arr, word_dict, class_dict):
        log_probs = []
        word_keys = list(word_dict.keys())
        sentiment_keys = list(class_dict.keys())
        for j in range(0, len(arr)):
            log_probs_list = list(range(len(sentiment_keys) - 1))
            for k in range(0, len(sentiment_keys) - 1):
                log_prob = 0
                for i in range(0, len(arr[j]) - 1):
                    for l in range(0, len(word_keys)):
                        if arr[j][i] + sentiment_keys[k] == word_keys[l]:
                            log_prob += math.log10(word_dict[word_keys[l]])
                            break
                        elif word_keys[l] == 'total':
                            log_prob += math.log10(word_dict['UNEXP' + sentiment_keys[k]])
                log_probs_list[k] = str(prior[k] + log_prob) + sentiment_keys[k]
            log_probs.append(log_probs_list)
        return log_probs
    
    #assigns a class based on which -log(prob) is larger
    def assign_class(self, class_arr):
        for j in range(0, len(class_arr)):
            max = -2147483648
            max_val = ''
            for i in range(0, len(class_arr[j])):
                if float(class_arr[j][i][:len(class_arr[j][i]) - 1]) > max:
                    max = float(class_arr[j][i][:len(class_arr[j][i]) - 1])
                    max_val = class_arr[j][i][len(class_arr[j][i]) - 1:len(class_arr[j][i])]
            class_arr[j] = max_val
        return class_arr

    #returns priors for each class and each line of the traning set, cleaned
    def preprocessing(self):
        csv_for_training = open(self.training_csv, 'r')
        sentiment_dict = {}
        word_dict = {}
        sum_dict = {}
        class_arr = []
        sum_arr = []
        vocab = 0

        #Create an array for each line
        training_arr = []
        for line in csv_for_training:
            arr = []
            arr.append(line)
            training_arr.append(arr)

        #create an array of only words, exclude dev sentences
        for j in range(0, len(training_arr)):
            for i in range(0, len(training_arr[j])):
                if j < len(training_arr) - 1 and j not in range(self.dev_num_low, self.dev_num_high):
                    training_arr[j][i] = training_arr[j][i][:len(training_arr[j][i]) - 1]
                    class_arr.append(training_arr[j][i][len(training_arr[j][i]) - 1:len(training_arr[j][i])])
                    training_arr[j][i] = training_arr[j][i][3:len(training_arr[j][i])]
                    training_arr[j][i] = training_arr[j][i].split(',')
                    sum_arr.append(len(training_arr[j][i]) - 1)
                elif j not in range(self.dev_num_low, self.dev_num_high):
                    class_arr.append(training_arr[j][i][len(training_arr[j][i]) - 1:len(training_arr[j][i])])
                    training_arr[j][i] = training_arr[j][i][3:len(training_arr[j][i])]
                    training_arr[j][i] = training_arr[j][i].split(',')
                    sum_arr.append(len(training_arr[j][i]) - 1)
                elif j in range(self.dev_num_low, self.dev_num_high):
                    training_arr[j][i] = training_arr[j][i][3:len(training_arr[j][i]) - 1]
                    training_arr[j][i] = training_arr[j][i].split(',')

        #3D to 2D training array
        training_arr = functools.reduce(operator.iconcat, training_arr, [])
        if self.model_type == 'binomial':
            #count if in each doc a word shows up
            binomial_arr = []
            for j in range(0, len(training_arr)):
                count_arr = []
                for i in range(0, len(training_arr[j]) - 1):
                    for k in range(i, len(training_arr[j])):
                        if Model.check_arr(count_arr, (training_arr[j][i])) == True:
                            break
                        elif training_arr[j][i] == training_arr[j][k] and j not in range(self.dev_num_low, self.dev_num_high):
                            count_arr.append(training_arr[j][i])
                if j not in range(self.dev_num_low, self.dev_num_high):
                    count_arr.append(training_arr[j][len(training_arr[j]) - 1])
                    binomial_arr.append(count_arr)
            word_dict = self.populate_word_dict(binomial_arr, word_dict)

        #creates dictionaries of each class, each word in each arrays sum, and each word with its corresponding class
        sentiment_dict = self.populate_class_dict(class_arr, sentiment_dict)
        sum_dict = self.populate_sum_dict(training_arr, sum_dict)
        if self.model_type == "standard":
            word_dict = self.populate_word_dict(training_arr, word_dict) 

        #add the total to each dictionary
        sentiment_dict['total'] = len(class_arr)
        values = list(word_dict.values())

        #calculates the total number of words in both classes
        sum = 0
        for i in range(0, len(values)):
            sum += values[i]
        sentiment_keys = list(sentiment_dict.keys())

        #deals with word we know in an unexpected class
        for i in range(0, len(sentiment_keys) - 1):
            word_dict['UNEXP' + sentiment_keys[i]] = 1
        word_dict['total'] = sum    

         #calculates the size of the vocabulary
        vocab = len(self.calculate_vocab(training_arr))
        return sentiment_dict, word_dict, vocab, sum_dict
    
    #Calculates priors and probs.
    def train(self):
        preprocessed = self.preprocessing()
        sentiment_dict = preprocessed[0]
        word_dict = preprocessed[1]
        vocab_size = preprocessed[2]
        sum_dict = preprocessed[3]
        sentiments = list(preprocessed[0].keys())
        sentiment_counts = list(preprocessed[0].values())
        words = list(preprocessed[1].keys())
        word_counts = list(preprocessed[1].values())

        if self.model_type == 'standard' or self.model_type == 'binomial':
            #Calculates priors for all classes
            for i in range(0, len(sentiments) - 1):
                sentiment_dict[sentiments[i]] = (sentiment_counts[i])/(sentiment_counts[len(sentiments) - 1])
            #Calculates probabilities for all classes
            for i in range(0, len(words) - 1):
                for j in range(0, len(sentiments) - 1):
                    #laplace smoothing for each prob
                    if words[i][len(words[i]) - 1:] == sentiments[j]:
                        word_dict[words[i]] = (word_counts[i] + 1)/(sum_dict[sentiments[j]] + vocab_size)
        else:
            print("You need to either make your model type standard or binomial.")
            return
        return sentiment_dict, word_dict
    
    #Finds out which model is best based on F1 scores
    @staticmethod
    def dev(model1, model2):
        standard = model1.train()
        binomial = model2.train()
        standard_priors = []
        binomial_priors = []
        standard_lines = open(model1.training_csv, 'r').readlines()
        binomial_lines = open(model2.training_csv, 'r').readlines()

        #Clean sentences and calculate priors for each
        standard_clean =  model1.calculate_priors(standard_lines, standard[0])
        binomial_clean = model2.calculate_priors(binomial_lines, binomial[0])
        standard_priors = standard_clean[0]
        binomial_priors = binomial_clean[0]
        standard_sentences = standard_clean[1]
        binomial_sentences = binomial_clean[1]
        
        #Calculate log probs for each word in a sentence
        log_probs_standard = model1.calculate_probs(standard_priors, standard_sentences, standard[1], standard[0])
        log_probs_binomial = model2.calculate_probs(binomial_priors, binomial_sentences, binomial[1], binomial[0])
       
        #Assigns class to each based on their scores
        log_probs_standard = model1.assign_class(log_probs_standard)
        log_probs_binomial = model2.assign_class(log_probs_binomial)

        sentiment_keys = list(standard[0].keys())

        #if data has more than two classes, return the dev classes
        if len(sentiment_keys) > 3:
            return log_probs_standard, log_probs_binomial
        
        #returns best model based on f1 score, if it can be calculated (in my case there is only one dev doc of one class, so it can't)
        else:
            precision_1 = model1.calculate_precision(standard_sentences,log_probs_standard, sentiment_keys)
            precision_2 = model2.calculate_precision(binomial_sentences, log_probs_binomial, sentiment_keys)
            recall_1 = model1.calculate_recall(standard_sentences,log_probs_standard, sentiment_keys)
            recall_2 = model2.calculate_recall(binomial_sentences, log_probs_binomial, sentiment_keys)
            try:
                f11 = (2*precision_1*recall_1)/(precision_1 + recall_1)
                f12 = (2*precision_2*recall_2)/(precision_2 + recall_2)
                if f11 > f12:
                    return "standard"
                elif f12 > f11:
                    return "binomial"
                else:
                    return "binomial"
            except:
                return "binomial"
    
    #Finds the class of the test set
    def test(self):
        model = self.train()
        priors = []
        lines = open(self.test_csv, 'r').readlines()

        #Clean sentences and calculate priors for each
        sentences = lines
        for i in range(0, len(sentences)):
            sentences[i] = sentences[i][3:len(sentences[i]) - 1]
            for key in model[0]:
                priors.append(math.log10(model[0][key]))
                              
        #Split each sentence into an array for processing
        for i in range(0, len(sentences)):
            sentences[i] = sentences[i].split(',')

        #Calculate log probs for each word in a sentence
        log_probs = self.calculate_probs(priors, sentences, model[1], model[0])

        #Assigns class to each based on their scores
        log_probs = self.assign_class(log_probs)

        return log_probs
