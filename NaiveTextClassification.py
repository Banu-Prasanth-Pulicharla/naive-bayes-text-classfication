import os
import math
from nltk.stem.snowball import SnowballStemmer

punctuation_str = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
                  'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stemmer = SnowballStemmer("english")


def processing_folder(folder):
    folder = folder.replace("\\", "/")
    list_of_files = os.listdir(folder)
    return list_of_files, folder


def file_list_to_dict(list_of_files, folder):
    email_dict = {}
    for file in list_of_files:
        path = folder+"/"+file

        with open(path, 'r', errors='ignore') as f:
            email_dict[file] = f.read()
    return email_dict


def dict_to_wordcount_dict(email_dict, ignore_stopwords):
    dict_with_word_count = {}
    for email in email_dict:
        email_dict[email] = email_dict[email].split()
        word_list = [''.join(temp for temp in word if temp not in punctuation_str)
                     for word in email_dict[email]]
        word_list = [temp for temp in word_list if temp]

        temp_word_list = []
        for word in word_list:
            temp_word_list.append(stemmer.stem(word))
        if ignore_stopwords == True:
            for sw in stopwords_list:
                if sw in temp_word_list:
                    temp_word_list.remove(sw)

        for word in temp_word_list:
            if word in dict_with_word_count:
                dict_with_word_count[word] = dict_with_word_count[word] + 1
            else:
                dict_with_word_count[word] = 1
    return dict_with_word_count


def calc_prob_given_class(train_spam_dict_count, train_ham_dict_count):
    train_spam_prob_dict = {}
    train_ham_prob_dict = {}

    total_spam_count = 0
    for key in train_spam_dict_count:
        total_spam_count += train_spam_dict_count[key]

    total_ham_count = 0
    for key in train_ham_dict_count:
        total_ham_count += train_ham_dict_count[key]

    dist_words = []
    for key in train_spam_dict_count:
        if key not in dist_words:
            dist_words.append(key)

    for key in train_ham_dict_count:
        if key not in dist_words:
            dist_words.append(key)
    total_words = len(dist_words)

    for key in train_spam_dict_count:
        val = 0
        val = (train_spam_dict_count[key] + 1) / \
            (total_spam_count + total_words)
        train_spam_prob_dict[key] = val

    for key in train_ham_dict_count:
        val = 0
        if key not in train_spam_prob_dict:
            val = 1 / (total_spam_count + total_words)
            train_spam_prob_dict[key] = val

    for key in train_ham_dict_count:
        val = 0

        val = (train_ham_dict_count[key] + 1) / \
            (total_ham_count + total_words)

        train_ham_prob_dict[key] = val

    for key in train_spam_dict_count:
        val = 0

        if key not in train_ham_prob_dict:
            val = 1 / (total_ham_count + total_words)

            train_ham_prob_dict[key] = val

    return train_spam_prob_dict, train_ham_prob_dict


def process_each_file(inp_folder, ham_prob_dict, spam_prob_dict, prior_ham, prior_spam, pred_token, ignore_stopwords):
    file_list, inp_folder = processing_folder(
        inp_folder)
    email_dict = file_list_to_dict(
        file_list, inp_folder)

    for email in email_dict:
        email_dict[email] = email_dict[email].split()
        word_list = [''.join(temp for temp in word if temp not in punctuation_str)
                     for word in email_dict[email]]
        word_list = [temp for temp in word_list if temp]
        temp_word_list = []
        for word in word_list:
            temp_word_list.append(stemmer.stem(word))

        if ignore_stopwords == True:
            for sw in stopwords_list:
                if sw in temp_word_list:
                    temp_word_list.remove(sw)

        email_dict[email] = temp_word_list
        ham_val = math.log2(prior_ham)
        for word in temp_word_list:
            if word in ham_prob_dict:
                ham_val = ham_val + math.log2(ham_prob_dict[word])

        spam_val = math.log2(prior_spam)
        for word in temp_word_list:

            if word in spam_prob_dict:
                spam_val = spam_val + math.log2(spam_prob_dict[word])

        if ham_val > spam_val:
            email_dict[email] = "HAM"
        elif ham_val < spam_val:
            email_dict[email] = "SPAM"
        else:
            # Defaulting to Ham if both value are same
            email_dict[email] = "HAM"

    counter = 0
    for file in email_dict:
        if email_dict[file] == pred_token:
            counter += 1

    return counter, len(email_dict)


input_value = input(
    "Enter the inputs with spaces: <Training Set Ham Path> <Training Set Spam Path> <Test Set Ham Path> <Test Set Spam Path>: ")
input_value = input_value.split(' ')


def main(input_value, ignore_stopwords):
    train_ham_inp_folder = input_value[0]
    train_spam_inp_folder = input_value[1]
    test_ham_inp_folder = input_value[2]
    test_spam_inp_folder = input_value[3]

    # Get Ham Data
    train_ham_file_list, train_ham_inp_folder = processing_folder(
        train_ham_inp_folder)
    train_ham_email_dict = file_list_to_dict(
        train_ham_file_list, train_ham_inp_folder)

    train_ham_dict_count = dict_to_wordcount_dict(
        train_ham_email_dict, ignore_stopwords)

    # Get Spam Data
    train_spam_file_list, train_spam_inp_folder = processing_folder(
        train_spam_inp_folder)

    train_spam_email_dict = file_list_to_dict(
        train_spam_file_list, train_spam_inp_folder)

    train_spam_dict_count = dict_to_wordcount_dict(
        train_spam_email_dict, ignore_stopwords)

    train_spam_prob_dict, train_ham_prob_dict = calc_prob_given_class(
        train_spam_dict_count, train_ham_dict_count)

    prior_train_spam = (len(train_spam_file_list)) / \
        (len(train_ham_file_list) + len(train_spam_file_list))
    prior_train_ham = (len(train_ham_file_list)) / \
        (len(train_ham_file_list) + len(train_spam_file_list))

    train_ham_ct, train_ham_ln = process_each_file(input_value[0], train_ham_prob_dict,
                                                   train_spam_prob_dict, prior_train_ham, prior_train_spam, "HAM", ignore_stopwords)
    train_spam_ct, train_spam_ln = process_each_file(input_value[1], train_ham_prob_dict,
                                                     train_spam_prob_dict, prior_train_ham, prior_train_spam, "SPAM", ignore_stopwords)
    test_ham_ct, test_ham_ln = process_each_file(input_value[2], train_ham_prob_dict,
                                                 train_spam_prob_dict, prior_train_ham, prior_train_spam, "HAM", ignore_stopwords)
    test_spam_ct, test_spam_ln = process_each_file(input_value[3], train_ham_prob_dict,
                                                   train_spam_prob_dict, prior_train_ham, prior_train_spam, "SPAM", ignore_stopwords)

    print("-----------------------")
    if ignore_stopwords == True:
        print("Without StopWords:")
    else:
        print("With StopWords:")
    print("-----------------------")
    print("Training Accuracy - " +
          str(round((train_ham_ct+train_spam_ct)/(train_ham_ln+train_spam_ln), 2)))
    print("Testing Accuracy - " +
          str(round((test_ham_ct+test_spam_ct)/(test_ham_ln+test_spam_ln), 2)))


# D:\data_TEMP\2\train\ham D:\data_TEMP\2\train\spam D:\data_TEMP\2\test\ham D:\data_TEMP\2\test\spam
main(input_value, False)
main(input_value, True)
