from superchunk_reader import *
from nltk.corpus.reader import *
from nltk.tokenize import RegexpTokenizer
from nltk import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
from nltk.metrics import *
from pprint import pprint

training_ids \
    = [1,2,6,7,9,13,14,27,55,67,70,74,76,80,81,84,96,115,131,147,153,154]
test_ids = [23,54,57,63,66,68,69,84,124,197]

def ids2iob_tupls(wsj_id_list):
    tupls = []
    for id in wsj_id_list:
        padded_id = str(id).zfill(4)
        wsj_text = open("../data/wsj_%s.pos" % padded_id).read()
        wsj_tree = superchunk2tree(wsj_text)
        pos_tupls = [tupl for tupl in tree2iob(wsj_tree)]
        for i, tupl in enumerate(pos_tupls):
            if i > 1:
                prev_tupl = pos_tupls[i - 1]
                pprev_tupl = pos_tupls[i - 1]
            elif i > 0:
                prev_tupl = pos_tupls[i - 1]
                pprev_tupl = None
            else: 
                prev_tupl = None
                pprev_tupl = None
            if i < len(pos_tupls) - 2:
                next_tupl = pos_tupls[i + 1]
                nnext_tupl = pos_tupls[i + 2]
            elif i < len(pos_tupls) - 1:
                next_tupl = pos_tupls[i + 1]
                nnext_tupl = None
            else: 
                next_tupl = None
                nnext_tupl = None
            tupls += (aug_features(tupl, prev_tupl, next_tupl, 
                                   pprev_tupl, nnext_tupl),)
    return tupls    

def aug_features(tupl, prev_tupl, next_tupl, pprev_tupl, nnext_tupl):
    iob_list = list(tupl)
    iob_list += [prev_tupl[1]] if prev_tupl else [""]  
    iob_list += [next_tupl[1]] if next_tupl else [""]
    iob_list += [prev_tupl[0]] if prev_tupl else [""]  
    iob_list += [next_tupl[0]] if next_tupl else [""]
    iob_list += [pprev_tupl[1]] if pprev_tupl else [""]  
    iob_list += [nnext_tupl[1]] if nnext_tupl else [""]
    return tuple(iob_list)

def tupls2classifier_format(tupls, is_test = False):
    classifier_format = []
    for tupl in tupls:
        features = ["tok", "pos", "np", "prev_pos", "next_pos", "prev_tok",\
                    "next_tok", "pprev_pos", "nnext_pos"]
        values = tupl[0:3] + tupl[4:6]
        feature_vector = dict(zip(features, values))
        if is_test:
            classifier_format.append(feature_vector)
        else:
            classifier_format.append((feature_vector, tupl[3],))
    return classifier_format

def classify(classifier, test, labels):
    test_results = dict(zip(labels, [set(), set(), set()]))
    for i, feature_vector in enumerate(test):
        test_class = classifier.classify(feature_vector)
        test_results[test_class].add(i)
    return test_results

def get_iob_tupls(training_ids, test_ids):
    iob_tupls = ids2iob_tupls(training_ids)
    test_tupls = ids2iob_tupls(test_ids)
    return (iob_tupls, test_tupls)

iob_tupls, test_tupls = get_iob_tupls(training_ids, test_ids)
classifer_format_test = tupls2classifier_format(test_tupls, True)
classifer_format_training = tupls2classifier_format(iob_tupls)

classifiers = {}
classifiers['bayes'] = NaiveBayesClassifier.train(classifer_format_training)
classifiers['maxent'] = MaxentClassifier.train(classifer_format_training)

labels = ["B-SNP", "I-SNP", "O"]
test_goldstd = dict(zip(labels, [set(), set(), set()]))
for i, iob_tupl in enumerate(ids2iob_tupls(test_ids)):
    test_goldstd[iob_tupl[3]].add(i)
for clsf_type, classifier in classifiers.iteritems():
    test_results = classify(classifier, classifer_format_test, labels)
    print 'B precision (%s):' % clsf_type, precision(test_goldstd["B-SNP"], 
                                                     test_results['B-SNP'])
    print 'B recall (%s):' % clsf_type, recall(test_goldstd["B-SNP"], 
                                               test_results['B-SNP'])
    print 'I precision (%s):' % clsf_type, precision(test_goldstd["I-SNP"], 
                                                     test_results['I-SNP'])
    print 'I recall (%s):' % clsf_type, recall(test_goldstd["I-SNP"], 
                                               test_results['I-SNP'])
    print 'O precision (%s):' % clsf_type, precision(test_goldstd["O"], 
                                                     test_results['O'])
    print 'O recall (%s):' % clsf_type, recall(test_goldstd["O"], 
                                               test_results['O'])