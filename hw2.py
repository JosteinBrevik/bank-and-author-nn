from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import csv
import statistics
import re

'''
This code is used to take in data from two csv datasets, preprocess this data and use it with sklearn's 
neural network to predict further results
Bank Management dataset: From UCI, contains info on people and their response to a marketing campaign
Spooky Author Identification dataset: Contains sentences from books by three different authors
Best results so far on a 80/20 training/testing split for both are 72,65% for BM and 99,97% for SAI
'''

'''
Fetches data from a file, then preprocesses each value according to its type
link: url to the data set
separator: used when the separator is not a comma (BM uses semicolon)
column_status: an array of integers that decides how the algorithm treats that column in the info
Final row (results) should always be set to 0
    0: Ignore it, that value is not used in the neural network
    1: Numerical, use the data as-is
    2: Categorical, use one-hot encoding before passing on
    3: Sentence, use this to split a sentence into words and create an "n-hot"-array. See the divide-and-conquer func
'''


class ANNCreator:
    def __init__(self, link, column_status, separator=','):
        self.columnStatus = column_status  # 0 = ignore, 1 = numerical, 2 = needs one-hot, 3 = split into words
        self.data = self.get_data(link, separator)
        self.relevantData = self.get_relevant_data()

    # Simply retrieves the data from a .csv file, and passes it on without the header
    def get_data(self, link, separator):
        data = []
        with open(link, newline='', encoding='utf-8') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=separator)
            for row in spamreader:
                if row:
                    data.append(row)
        print("Categories:")
        print(data[0])
        print("First value:")
        print(data[1])
        return data[1:]

    # Gets all the data for one category, so that it can be processed together. Ignores columns marked '0'
    # Could probably be one-lined
    def get_columns(self):
        columns = []
        data = self.data
        for i in range(len(data[0])):
            if self.columnStatus[i]:
                columns.append([row[i] for row in data])
            else:
                columns.append([])  # Append empty so that get_relevant_data works

        return columns

    # Creates a one-hot array from the data (array with (N = unique values in column) values, where the one that
    # matches that rows category set to 1 and the others to 0)
    # Uses a set in the beginning to get all unique values, then runs through and sets correct 1's
    # Effective with well-defined categories (married, divorced, single), less so when each value is different
    def one_hot_array(self, in_array):
        categories = set()
        for i in in_array:
            categories.add(i)

        categories = list(categories)
        categories.sort()  # sort to always have same order
        size = len(categories)

        out_array = []
        for category in in_array:
            out_array.append([1 if category == categories[j] else 0 for j in range(size)])  # <3 one-liners

        return out_array

    # Switch case for how to treat each column
    def get_relevant_data(self):
        columns = self.get_columns()
        relevant = []
        for i in range(len(columns)):
            status = self.columnStatus[i]
            if status == 0:
                continue
            elif status == 1:
                relevant.append([[float(x)] for x in columns[i]])
            elif status == 2:
                relevant.append(self.one_hot_array(columns[i]))
            elif status == 3:
                relevant.append(self.divide_a_sentence_and_conquer(columns[i]))
        return relevant

    # Returns the processed data for one row
    def get_relevant_row_data(self, index):
        personal_data = []
        for category in self.relevantData:
            personal_data += category[index]
        return personal_data

    # Returns the processed data for all rows
    def create_complete_array(self):
        out_array = []
        for i in range(len(self.data)):
            out_array.append(self.get_relevant_row_data(i))
        return out_array

    # Returns the results column
    # Assumes the last column is the result we're after. If not, there will be trouble
    def get_results(self):
        return [x[-1] for x in self.data]

    '''
    Creates an "n-hot" array of all words used in all sentences, as long as that word appears a set amount of times
    Turns each sentence into an array that displays how many times each word appears in that sentence
    Threshold unit decides how many times a word must be found to be counted. Also avoids MemoryError
    Testing shows that a high threshold provides better results
    :sentence_col: The column containing sentences (passed in from get_relevant_data)
    
    Improvement ideas: Implement upper threshold
                       Use find_word_deviation and another threshold to only act on words that give good info
    '''

    def divide_a_sentence_and_conquer(self, sentence_col):
        words = {}
        sentences = []

        # Split each sentence into an array of lowercase words
        for sentence in sentence_col:
            words_in_sentence = re.sub("[^\w]", " ", sentence).split()
            words_in_sentence = [x.lower() for x in words_in_sentence]
            sentences.append(words_in_sentence)

        # Count how often each word appears in total
        for sentence in sentences:
            for word in sentence:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

        # Remove words under the threshold, then create a set for more efficient lookup
        threshold = 100
        print([[word, words[word]] for word in words.keys() if words[word] > threshold])
        legit_words = [word for word in words.keys() if words[word] > threshold]
        word_set = set(legit_words)
        size = len(legit_words)
        print("Num of words: ", size)

        # For each sentence, create an array showing how often each of its words appear (if word is above threshold)
        out_array = []
        for sentence in sentences:
            n_hot_array = [0] * size
            for word in sentence:
                if word in word_set:
                    n_hot_array[legit_words.index(word)] += 1

            out_array.append(n_hot_array)

        return out_array


'''
Ad hoc code for finding the standard deviation of word use by the different authors
word_scores is copy-pasted from the output of divide_a_sentence_and_conquer
Runs through the data looking for each of the words, and counts how often each author used it
This data is then run through normalization and the standard deviation is found
Some data is printed in order to provide example data for the report I had to write for this whole thing
'''
def find_word_deviation():
    data = []
    with open('train2.csv', newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row:
                data.append(row)

    word_scores = [['me', 456], ['means', 168], ['of', 146], ['might', 133], ['set', 153], ['out', 104], ['fact', 159],
                   ['seemed', 544], ['wall', 115], ['be', 118], ['left', 141], ['hand', 292], ['was', 151],
                   ['which', 367], ['manner', 163], ['took', 227], ['air', 240], ['possible', 143], ['self', 103],
                   ['is', 108], ['looked', 231], ['heart', 153], ['not', 125], ['countenance', 128], ['youth', 104],
                   ['passed', 219], ['best', 122], ['gentle', 112], ['character', 130], ['cannot', 182], ['heard', 158],
                   ['him', 363], ['felt', 343], ['myself', 378], ['body', 228], ['knew', 105], ['brought', 170],
                   ['subject', 102], ['had', 114], ['eyes', 540], ['find', 225], ['will', 133], ['idea', 195],
                   ['gave', 238], ['course', 214], ['own', 492], ['doubt', 118], ['became', 295], ['been', 1430],
                   ['too', 172], ['expression', 111], ['give', 106], ['held', 114], ['followed', 104], ['taken', 168],
                   ['place', 196], ['said', 104], ['known', 186], ['made', 123], ['kind', 185], ['us', 596],
                   ['power', 169], ['themselves', 105], ['view', 130], ['secret', 103], ['off', 247], ['than', 217],
                   ['less', 109], ['peculiar', 101], ['moment', 259], ['months', 101], ['sky', 115], ['trees', 134],
                   ['seen', 282], ['grew', 146], ['fell', 157], ['turned', 181], ['reached', 112], ['away', 215],
                   ['replied', 119], ['person', 160], ['longer', 152], ['came', 461], ['gone', 121], ['ancient', 120],
                   ['stood', 163], ['itself', 203], ['head', 113], ['form', 187], ['appeared', 175], ['words', 170],
                   ['case', 136], ['rather', 138], ['word', 136], ['change', 136], ['lips', 102], ['thing', 150],
                   ['observed', 109], ['hands', 159], ['length', 149], ['door', 303], ['herself', 104], ['care', 103],
                   ['remained', 138], ['call', 110], ['together', 126], ['began', 187], ['himself', 184], ['saw', 349],
                   ['given', 127], ['discovered', 111], ['open', 196], ['small', 153], ['whom', 221], ['called', 173],
                   ['moon', 143], ['existence', 117], ['sound', 116], ['short', 111], ['way', 135], ['scene', 169],
                   ['die', 103], ['led', 122], ['whose', 153], ['thought', 345], ['purpose', 102], ['greater', 103],
                   ['voice', 271], ['portion', 105], ['alone', 199], ['floor', 137], ['mind', 378], ['enough', 110],
                   ['attention', 137], ['right', 120], ['cause', 102], ['living', 107], ['feelings', 116],
                   ['period', 136], ['become', 171], ['died', 127], ['done', 145], ['account', 126],
                   ['appearance', 131], ['wish', 111], ['sight', 162], ['merely', 113], ['arms', 102], ['feeling', 115],
                   ['wind', 137], ['chamber', 106]]

    max_dev = 0
    min_dev = 1
    for word_score in word_scores:
        word = word_score[0]
        author_scores = {'EAP': 0, 'HPL': 0, 'MWS': 0}
        for i in data:
            if word in i[1]:
                author_scores[i[2]] += 1
        normalized = [float(i) / sum(author_scores.values()) for i in author_scores.values()]
        deviation = statistics.stdev(normalized)
        if deviation > max_dev:
            max_dev = deviation
            max_dev_word = word
            print(max_dev_word, max_dev, author_scores)
        if deviation < min_dev:
            min_dev = deviation
            min_dev_word = word
            print(min_dev_word, min_dev, author_scores)


# Tests how well the prediction went. Hardcoded to look for "yes" or "no", so if other results are used this must
# be changed. Also outputs a confusion matrix
def get_hit_rate(prediction, correct):
    score = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    num = len(prediction)
    for i in range(num):
        if prediction[i] == correct[i]:
            score += 1
            if prediction[i] == 'yes':
                true_pos += 1
            else:
                true_neg += 1
        elif prediction[i] == 'yes':
            false_pos += 1
        else:
            false_neg += 1

    print(
        "Hit %s out if %s, %s percent correct. \n%s true positives\n%s true negatives\n%s false positives\n%s false negatives" % (
            score, num, score / num, true_pos, true_neg, false_pos, false_neg))


# Same as above, but for the authors instead. Also not usable for anything other than this exact data
# Could join these two together using arrays and input parameters. Would also look better
def get_author_rate(prediction, correct):
    score = 0
    true_eap = true_hpl = 0
    true_mws = 0
    false_eap = 0
    false_hpl = 0
    false_mws = 0
    num = len(prediction)
    wrong_predictions = []
    for i in range(num):
        if prediction[i] == correct[i]:
            score += 1
            if prediction[i] == 'EAP':
                true_eap += 1
            elif prediction[i] == 'HPL':
                true_hpl += 1
            else:
                true_mws += 1
        else:
            wrong_predictions.append(str(i) + " " + prediction[i])
            if prediction[i] == 'EAP':
                false_eap += 1
            elif prediction[i] == 'HPL':
                false_hpl += 1
            else:
                false_mws += 1

    print("Wrong predictions: ", wrong_predictions)  # Nice to see where it failed
    print(
        "Hit %s out if %s, %s percent correct. \n%s true EAP\n%s true HPL\n%s true MWS\n%s false EAP\n%s false HPL\n%s false MWS" % (
            score, num, score * 100 / num, true_eap, true_hpl, true_mws, false_eap, false_hpl, false_mws))

# Sets the cutoff for the data (80% on both) and fetches the correct data
# Creates the network using sklearn's scaler and MLPClassifier
def create_network(data_type):
    cutoff = 0
    ac = None

    if data_type == 'bank':
        cutoff = 36000
        #                                                         | = loan
        ac = ANNCreator('bank-full.csv', [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 2, 0], ';')
        #  to use:                       [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 0])
        # "age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"
    elif data_type == 'author':
        cutoff = 11200
        ac = ANNCreator('train2.csv', [0, 3, 2])

    X = ac.create_complete_array()[:cutoff]
    results = ac.get_results()[:cutoff]
    to_be_predicted = ac.create_complete_array()[cutoff:]

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    to_be_predicted = scaler.transform(to_be_predicted)

    clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(200, 50), random_state=1)
    clf.fit(X, results)

    if data_type == 'bank':
        get_hit_rate(clf.predict(to_be_predicted), ac.get_results()[cutoff:])
    elif data_type == 'author':
        get_author_rate(clf.predict(to_be_predicted), ac.get_results()[cutoff:])


create_network('author')
