from gensim import corpora, models
from stop_words import get_stop_words
from pprint import pprint
from pathlib import Path

CATEGORIES = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
              'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.gun', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
EN_STOP = get_stop_words('en')
PARENT = "document-classification"

# Train LDA Model
def train_lda(datasetpath):
    documents = []
    f = open(f'{datasetpath}/20ng-train-stemmed.txt')
    for line in f:
        if line:
            parts = line.split()
            documents.append([v.strip()
                             for v in parts[1:] if v and v not in EN_STOP])
    f.close()

    D = corpora.Dictionary(documents)
    # print(D)
    # each tokens converts into numeric id and its frequency in a document
    C = [D.doc2bow(doc) for doc in documents]
    # print(C)

    lda = models.ldamodel.LdaModel(
        C, num_topics=len(CATEGORIES), id2word=D, passes=10)
    lda.save(f'{datasetpath}/trained_lda')

# Display 20 topics and word distribution
def show(datasetpath):
    lda = models.ldamodel.LdaModel.load(f'{datasetpath}/trained_lda')
    pprint(lda.show_topics(20))

# Predict on test dataset
def predict(datasetpath):
    f = open(f'{datasetpath}/20ng-test-stemmed.txt')
    cats = []
    documents = []

    for line in f:
        if line:
            parts = line.split()
            doc = [v.strip() for v in parts[1:] if v and v not in EN_STOP]
            if not doc:
                continue
            documents.append(doc)
            cats.append(parts[0])
    f.close()

    D = corpora.Dictionary(documents)
    C = [D.doc2bow(document) for document in documents]
    lda = models.ldamodel.LdaModel.load(f'{datasetpath}/trained_lda')

    # Evaluation of LDA
    matched = 0
    for i, doc in enumerate(C[:10]):
        for i in range(len(doc)):
            pprint("Word {} (\"{}\") appears {} time.".format(doc[i][0],
                                                              D[doc[i][0]],
                                                              doc[i][1]))
        #topics = lda.get_document_topics(D.doc2bow(doc))
        topics = lda[doc]
        # pprint.pprint(topics)
        estimate = max(topics, key=lambda x: x[1])
        estimated_topic = CATEGORIES[estimate[0]]
        #print(f'document: {documents[i]}')
        pprint(f'real topics: {cats[i]}')
        pprint(f'estimated topic: {estimated_topic} {estimate}')
        pprint('---------------------------------')
        if cats[i] == estimated_topic:
            matched += 1
    # Total Accuracy on 10 documents
    acc = (matched/10) * 100
    pprint(f'Model Accuracy on 10 documents: {acc}')


if __name__ == '__main__':
    data_path = Path(PARENT).parent / "../10_source_files"
    train_lda(data_path)
    show(data_path)
    predict(data_path)
