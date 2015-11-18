import glob, math, os, re, string, sys,copy,operator
from collections import OrderedDict
from operator import itemgetter

class PreProcess:

    def __init__(self): pass

    ## Function to print the confusion matrix.
    ## Argument 1: "actual" is a list of integer class labels, one for each test example.
    ## Argument 2: "predicted" is a list of integer class labels, one for each test example.
    ## "actual" is the list of actual (ground truth) labels.
    ## "predicted" is the list of labels predicted by your classifier.
    ## "actual" and "predicted" MUST be in one-to-one correspondence.
    ## That is, actual[i] and predicted[i] stand for testfile[i].
    def printConfMat(self, actual, predicted):
        all_labels = sorted(set(actual + predicted))
        assert(len(actual) == len(predicted))
        confmat = {}  ## Confusion Matrix
        for i,a in enumerate(actual): confmat[(a, predicted[i])] = confmat.get((a, predicted[i]), 0) + 1
        print
        print
        print "0",  ## Actual labels column (aka first column)
        for label2 in all_labels:
            print label2,
        print
        for label in all_labels:
            print label,
            for label2 in all_labels:
                print confmat.get((label, label2), 0),
            print

    ## Function to remove leading, trailing, and extra space from a string.
    ## Inputs a string with extra spaces.
    ## Outputs a string with no extra spaces.
    def remove_extra_space(self, input_string):
        return re.sub("\s+", " ", input_string.strip())

    ## Tokenizer.
    ## Input: string
    ## Output: list of lowercased words from the string
    def word_tokenize(self, input_string):
        extra_space_removed = self.remove_extra_space(input_string)
        punctuation_removed = "".join([x for x in extra_space_removed if x not in string.punctuation])
        lowercased = punctuation_removed.lower()
        return lowercased.split()


class BernoulliNaiveBayes:
    def __init__(self, train_test_dir, vocab, p = None):
        self.train_test_dir = train_test_dir
        self.vocab = vocab
        self.p = p

    ## Define Train function
    def train(self):
        x = PreProcess()
        f = open(vocab,'r')
        stopwords = f.read()
        f.close()
        stopwords = x.word_tokenize(stopwords)

        if train_test_dir == 'AAAC_problems/problemA/':
            Na = 13
        elif train_test_dir == 'AAAC_problems/problemB/':
            Na = 13
        elif train_test_dir == 'AAAC_problems/problemC/':
            Na = 5
        elif train_test_dir == 'AAAC_problems/problemG/':
            Na = 2
        elif train_test_dir == 'AAAC_problems/problemH/':
            Na = 3

        files = os.listdir(train_test_dir)
        os.chdir(train_test_dir)
        

        prior = []
        authors = []
        condprob = []
        b = []
        for i in range(0,Na):
            c = copy.deepcopy(b)
            authors.append(c)

        i = 0

        for article in files:
            if article == '.DS_Store':
                continue
            elif "sample" in article:
                continue
            elif "train01" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[0].append(article_new)
                i = i+1
            elif "train02" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[1].append(article_new)
                i = i+1
            elif "train03" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[2].append(article_new)
                i = i+1
            elif "train04" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[3].append(article_new)
                i = i+1
            elif "train05" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[4].append(article_new)
                i = i+1
            elif "train06" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[5].append(article_new)
                i = i+1
            elif "train07" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[6].append(article_new)
                i = i+1
            elif "train08" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[7].append(article_new)
                i = i+1
            elif "train09" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[8].append(article_new)
                i = i+1
            elif "train10" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[9].append(article_new)
                i = i+1
            elif "train11" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[10].append(article_new)
                i = i+1
            elif "train12" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[11].append(article_new)
                i = i+1
            elif "train13" in article:
                f = open(article,'r')
                article_new = f.read()
                f.close()
                article_new = x.word_tokenize(article_new)
                authors[12].append(article_new)
                i = i+1

        N = i

        j = 0

        new_stopwords = {}

        for word in stopwords:
            new_stopwords[word] = 0

        #print sorted(new_stopwords.items(), key=operator.itemgetter(1))


        for c in authors:
            for ca in c:
                for oneword in ca:
                    if oneword in stopwords:
                        new_stopwords[oneword] = new_stopwords[oneword]+1
        new_stopwords_list =  sorted(new_stopwords,key = new_stopwords.get,reverse = True)

        new_stopwords_list = new_stopwords_list[:423]
        #print new_stopwords_list
        #print len(new_stopwords_list)

        #stopwords = new_stopwords_list

        for c in authors:
            diction = {}
            Nc = len(c)
            prior.append(Nc*1.0/N)
            for v in stopwords:
                Nci = 0
                rec = 0
                for ca in c:
                    if v in ca:
                        Nci = Nci+1
                diction[v] = (Nci+1)*1.0/(Nc+2)
            condprob.append(diction)


        j = 0
        CCEi = {}
        for v in stopwords:
            summ = 0
            j = 0
            for c in authors:
                summ = summ - prior[j]*condprob[j][v]*math.log(condprob[j][v],2)
                j = j+1
            CCEi[v] = summ

        print OrderedDict(sorted(CCEi.items(),key =itemgetter(1),reverse = True))

        return stopwords,prior,condprob,new_stopwords_list
    #Define Test function
    def test(self, prior, condprob, all_labels, testfilename):
        x = PreProcess()
        i = 0
        f = open(testfilename,'r')
        testfile = f.read()
        f.close()
        testfile = x.word_tokenize(testfile)
        score = []
        for author in prior:
            summ = 0
            summ = summ + math.log(prior[i],2)
            for v in all_labels:
                if v in testfile:
                    summ = summ + math.log(condprob[i][v],2)
                else:
                    summ = summ + math.log(1-condprob[i][v],2)
            score.append(summ)
            i = i+1
        return score.index(max(score))+1



train_test_dir = sys.argv[1]
ground_truth = []

if train_test_dir == 'AAAC_problems/problemA/':
    ground_truth = [3,13,11,7,10,12,8,1,5,4,6,2,9]
elif train_test_dir == 'AAAC_problems/problemB/':
    ground_truth = [1,9,13,5,10,3,7,8,11,12,2,6,4]
elif train_test_dir == 'AAAC_problems/problemC/':
    ground_truth = [3,1,1,4,5,2,4,5,2]
elif train_test_dir == 'AAAC_problems/problemG/':
    ground_truth = [2,1,2,1]
elif train_test_dir == 'AAAC_problems/problemH/':
    ground_truth = [3,1,2]

vocab = 'stopwords.txt'


x = BernoulliNaiveBayes(train_test_dir, vocab, None)
[a,b,c,a1] = x.train()

files = os.listdir(os.getcwd())
sampleNames = []
for article in files:
    if article == '.DS_Store':
        continue
    elif "train" in article:
        continue
    elif "sample" in article:
        sampleNames.append(article)

res = []

for testfilename in sampleNames:
    res.append(x.test(b,c,a,testfilename))
#print res
#print ground_truth
correct = 0
ind = 0

print 'Accuracy:'

for r in res:
    if r == ground_truth[ind]:
        correct = correct+1
    ind = ind+1

print correct*1.0/len(res)

print 'Confusion matrix:'

y = PreProcess()
y.printConfMat(ground_truth,res)


res = []
for i in range(1,43):
    j = i*10
    new_stopwords_list = a1[:i*10]
    res = []
    print 'Accuracy with first '+str(j)+' features'
    for testfilename in sampleNames:
        res.append(x.test(b,c,new_stopwords_list,testfilename))
    #print res
    #print ground_truth
    correct = 0
    ind = 0

    for r in res:
        if r == ground_truth[ind]:
            correct = correct+1
        ind = ind+1

    print correct*1.0/len(res)




