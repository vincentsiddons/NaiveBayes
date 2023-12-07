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
        #count the number of times each sentiment and word shows up
        if self.model_type == 'standard':
            for i in range(0, len(class_arr)):
                try:
                    sentiment_dict[class_arr[i]] += 1
                    sum_dict[class_arr[i]] += sum_arr[i]
                except:
                    sentiment_dict[class_arr[i]] = 1
                    sum_dict[class_arr[i]] = sum_arr[i]
            for j in range(0, len(training_arr)):
                for i in range(0, len(training_arr[j]) - 1):
                    try:
                        word_dict[training_arr[j][i] + training_arr[j][len(training_arr[j]) - 1]] += 1
                    except:
                        word_dict[training_arr[j][i] + training_arr[j][len(training_arr[j]) - 1]] = 1      
            vocab = len(list(word_dict.keys()))//2
            #add the total to each dictionary
            sentiment_dict['total'] = len(class_arr)
            values = list(word_dict.values())
            #not correct, needs to be class sum
            sum = 0
            for i in range(0, len(values)):
                sum += values[i]
            sentiment_keys = list(sentiment_dict.keys())
            #deals with word we know in an unexpected class
            for i in range(0, len(sentiment_keys) - 1):
                word_dict['UNEXP' + sentiment_keys[i]] = 1
            word_dict['total'] = sum
        elif self.model_type == 'binomial':
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
            training_arr = binomial_arr
            #count the number of times each sentiment and word shows up
            for i in range(0, len(class_arr)):
                try:
                    sentiment_dict[class_arr[i]] += 1
                    sum_dict[class_arr[i]] += sum_arr[i]
                except:
                    sentiment_dict[class_arr[i]] = 1
                    sum_dict[class_arr[i]] = sum_arr[i]
            for j in range(0, len(training_arr)):
                for i in range(0, len(training_arr[j]) - 1):
                    try:
                        word_dict[training_arr[j][i] + training_arr[j][len(training_arr[j]) - 1]] += 1
                    except:
                        word_dict[training_arr[j][i] + training_arr[j][len(training_arr[j]) - 1]] = 1      
            vocab = len(list(word_dict.keys()))//2
            #add the total to each dictionary
            sentiment_dict['total'] = len(class_arr)
            values = list(word_dict.values())
            #not correct, needs to be class sum
            sum = 0
            for i in range(0, len(values)):
                sum += values[i]
            sentiment_keys = list(sentiment_dict.keys())
            #deals with word we know in an unexpected class
            for i in range(0, len(sentiment_keys) - 1):
                word_dict['UNEXP' + sentiment_keys[i]] = 1
            word_dict['total'] = sum     
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
                    #laplace smoothing
                    if words[i][len(words[i]) - 1:] == sentiments[j]:
                        word_dict[words[i]] = (word_counts[i] + 1)/(sum_dict[sentiments[j]] + vocab_size)
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
        standard_sentences = standard_lines[model1.dev_num_low:model1.dev_num_high]
        for i in range(0, len(standard_sentences)):
            standard_sentences[i] = standard_sentences[i][3:len(standard_sentences[i]) - 1]
            for key in standard[0]:
                standard_priors.append(math.log10(standard[0][key]))
        binomial_sentences = binomial_lines[model2.dev_num_low:model2.dev_num_high]
        for i in range(0, len(binomial_sentences)):
            binomial_sentences[i] = binomial_sentences[i][3:len(binomial_sentences[i]) - 1]
            for key in binomial[0]:
                binomial_priors.append(math.log10(binomial[0][key]))

        #Split each sentence into an array for processing
        for i in range(0, len(standard_sentences)):
            standard_sentences[i] = standard_sentences[i].split(',')
        for i in range(0, len(binomial_sentences)):
            binomial_sentences[i] = binomial_sentences[i].split(',')
        
        #Calculate log probs for each word in a sentence
        #TO DO: Make this itself a function
        log_probs_standard = []
        log_probs_binomial = []
        word_keys = list(standard[1].keys())
        sentiment_keys = list(standard[0].keys())
        for j in range(0, len(standard_sentences)):
            log_probs_list = list(range(len(sentiment_keys) - 1))
            for k in range(0, len(sentiment_keys) - 1):
                log_prob = 0
                for i in range(0, len(standard_sentences[j]) - 1):
                #TO DO: Go through words dictionary, if match, log_probs[i] = math.log10(standard[1][key]),
                #elif at 'total' log_probs[i] = math.log10(standard[1]['UNEXP'+standard_sentences[j][len(standard_sentences[j]) - 1])])
                    for l in range(0, len(word_keys)):
                        #PROBLEM: no excpetions for those words in the wrong class
                        if standard_sentences[j][i] + sentiment_keys[k] == word_keys[l]:
                            log_prob += math.log10(standard[1][word_keys[l]])
                            break
                        elif word_keys[l] == 'total':
                            log_prob += math.log10(standard[1]['UNEXP'])
                log_probs_list[k] = str(standard_priors[k] + log_prob) + sentiment_keys[k]
            log_probs_standard.append(log_probs_list)

        for j in range(0, len(binomial_sentences)):
            log_probs_list = list(range(len(sentiment_keys) - 1))
            for k in range(0, len(sentiment_keys) - 1):
                log_prob = 0
                for i in range(0, len(binomial_sentences[j]) - 1):
                    for l in range(0, len(word_keys)):
                        if binomial_sentences[j][i] + sentiment_keys[k] == word_keys[l]:
                            log_prob += math.log10(binomial[1][word_keys[l]])
                            break
                        elif word_keys[l] == 'total':
                            log_prob += math.log10(binomial[1]['UNEXP'])
                log_probs_list[k] = str(standard_priors[k] + log_prob) + sentiment_keys[k]
            log_probs_binomial.append(log_probs_list)
        
        #Assigns class to each based on their scores
        for j in range(0, len(log_probs_standard)):
            max = -2147483648
            max_val = ''
            for i in range(0, len(log_probs_standard[j])):
                if float(log_probs_standard[j][i][:len(log_probs_standard[j][i]) - 1]) > max:
                    max = float(log_probs_standard[j][i][:len(log_probs_standard[j][i]) - 1])
                    max_val = log_probs_standard[j][i][len(log_probs_standard[j][i]) - 1:len(log_probs_standard[j][i])]
            log_probs_standard[j] = max_val

        for j in range(0, len(log_probs_binomial)):
            max = -2147483648
            max_val = ''
            for i in range(0, len(log_probs_binomial[j])):
                if float(log_probs_binomial[j][i][:len(log_probs_binomial[j][i]) - 1]) > max:
                    max = float(log_probs_binomial[j][i][:len(log_probs_binomial[j][i]) - 1])
                    max_val = log_probs_binomial[j][i][len(log_probs_binomial[j][i]) - 1:len(log_probs_binomial[j][i])]
            log_probs_binomial[j] = max_val

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
        #TO DO: Make this itself a function
        log_probs= []
        word_keys = list(model[1].keys())
        sentiment_keys = list(model[0].keys())
        for j in range(0, len(sentences)):
            log_probs_list = list(range(len(sentiment_keys) - 1))
            for k in range(0, len(sentiment_keys) - 1):
                log_prob = 0
                for i in range(0, len(sentences[j]) - 1):
                #TO DO: Go through words dictionary, if match, log_probs[i] = math.log10(standard[1][key]),
                #elif at 'total' log_probs[i] = math.log10(standard[1]['UNEXP'+standard_sentences[j][len(standard_sentences[j]) - 1])])
                    for l in range(0, len(word_keys)):
                        #PROBLEM: no excpetions for those words in the wrong class
                        if sentences[j][i] + sentiment_keys[k] == word_keys[l]:
                            log_prob += math.log10(model[1][word_keys[l]])
                            break
                log_probs_list[k] = str(priors[k] + log_prob) + sentiment_keys[k]
            log_probs.append(log_probs_list)
        
        #Assigns class to each based on their scores
        for j in range(0, len(log_probs)):
            max = -2147483648
            max_val = ''
            for i in range(0, len(log_probs[j])):
                if float(log_probs[j][i][:len(log_probs[j][i]) - 1]) > max:
                    max = float(log_probs[j][i][:len(log_probs[j][i]) - 1])
                    max_val = log_probs[j][i][len(log_probs[j][i]) - 1:len(log_probs[j][i])]
            log_probs[j] = max_val

        return log_probs


