from superchunk_reader import *
from nltk.corpus.reader import *
from nltk.tokenize import RegexpTokenizer
from nltk import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
from nltk.metrics import *

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

def tupls2classifier_format(tupls):
    classifier_format = []
    features = []
    for tupl in tupls:
        featureVector = {}
        featureVector["token_" + tupl[0]] = 1
        featureVector["pos_" + tupl[1]] = 1
#        featureVector["np_" + tupl[2]] = 1
#        featureVector["prev_pos_" + tupl[4]] = 1
#        featureVector["next_pos_" + tupl[5]] = 1
        featureVector["prev_token_" + tupl[6]] = 1
        featureVector["next_token_" + tupl[7]] = 1
#        featureVector["pprev_pos_" + tupl[8]] = 1
#        featureVector["nnext_pos_" + tupl[9]] = 1
        features += ["token_" + tupl[0],
                     "pos_" + tupl[1], "np_" + tupl[2], 
#                     "prev_pos_" + tupl[4], "next_pos_" + tupl[5],
                     "prev_token_" + tupl[6],#, "next_token_" + tupl[7],
#                     "pprev_token_" + tupl[8], "nnext_token_" + tupl[9]
                     ]
        classifier_format.append((featureVector, tupl[3],))
    features = set(features)
    features = dict(zip(features, [0]*len(features)))
    for i, f in enumerate(classifier_format):
        features_tmp = features.copy()
        features_tmp.update(classifier_format[i][0])
        classifier_format[i] = (features_tmp,classifier_format[i][1],)
    return classifier_format, features

def collect_pos(gen):
    features = set([tupl[1] for tupl in gen])
    return features

def collect_np(iob_tuples):
    features = set([tupl[2] for tupl in iob_tuples])
    return features

def getNaiveBayesClassifier(iobTuples):
    training, features = tupls2classifier_format(iobTuples)
    print len(features)
    print len(training)
    return NaiveBayesClassifier.train(training)
#    return MaxentClassifier.train(training)

def classify(classifier, test, labels):
    test_results = dict(zip(labels, [set(), set(), set()]))
    for i, featureVector in enumerate(test):
        test_class = classifier.classify(featureVector)
        test_results[test_class].add(i)
    return test_results

def get_iob_tuples(training_ids, test_ids):
    iob_tuples = ids2iob_tupls(training_ids)
    test_tuples = ids2iob_tupls(test_ids)
    return (iob_tuples, test_tuples)

(iob_tuples, test_tuples) = get_iob_tuples(training_ids, test_ids)
classifer_format, features = tupls2classifier_format(test_tuples)
test = [featureVector for (featureVector, label) in classifer_format]
classifier = getNaiveBayesClassifier(iob_tuples)
#labels = ["B-SNP", "I-SNP", "O"]

#test_results = classify(classifier, test, labels)
#test_goldstd = dict(zip(labels, [set(), set(), set()]))
#for i, iobTuple in enumerate(ids2iob_tupls(test_ids)):
#    test_goldstd[iobTuple[3]].add(i)
#
#print 'pos precision:', precision(test_goldstd["B-SNP"], test_results['B-SNP'])
#print 'pos recall:', recall(test_goldstd["B-SNP"], test_results['B-SNP'])
#print 'pos precision:', precision(test_goldstd["I-SNP"], test_results['I-SNP'])
#print 'pos recall:', recall(test_goldstd["I-SNP"], test_results['I-SNP'])
#print 'pos precision:', precision(test_goldstd["O"], test_results['O'])
#print 'pos recall:', recall(test_goldstd["O"], test_results['O'])