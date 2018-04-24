
# coding: utf-8

# # TensorBoard Visualizations
# 
# 
# In this tutorial, we will learn how to visualize different types of NLP based Embeddings via TensorBoard. TensorBoard is a data visualization framework for visualizing and inspecting the TensorFlow runs and graphs. We will use a built-in Tensorboard visualizer called *Embedding Projector* in this tutorial. It lets you interactively visualize and analyze high-dimensional data like embeddings.
# 

# ## Read Data 
# 
# For this tutorial, a transformed MovieLens dataset<sup>[1]</sup> is used. You can download the final prepared csv from [here](https://github.com/parulsethi/DocViz/blob/master/movie_plots.csv).

# In[49]:

import gensim
import pandas as pd
import smart_open
import random

# read data
dataframe = pd.read_csv('/Users/Shemelis/10-16-2017-GW-Arlington-Class-Repository-DATA/Machine-Learning-Project/All_Journal_data.csv',index_col='pmid')
# /Users/Shemelis/10-16-2017-GW-Arlington-Class-Repository-DATA/Machine-Learning-Project/All_Journal_data.csv
dataframe=dataframe. sample(n=10, axis=0)
dataframe.head()


# # 1. Visualizing Doc2Vec
# In this part, we will learn about visualizing Doc2Vec Embeddings aka [Paragraph Vectors](https://arxiv.org/abs/1405.4053) via TensorBoard. The input documents for training will be the synopsis of movies, on which Doc2Vec model is trained. 
# 
# <img src="Tensorboard.png">
# 
# The visualizations will be a scatterplot as seen in the above image, where each datapoint is labelled by the movie title and colored by it's corresponding genre. You can also visit this [Projector link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/DocViz/master/movie_plot_config.json) which is configured with my embeddings for the above mentioned dataset. 
# 
# 
# ## Preprocess Text

# Below, we define a function to read the training documents, pre-process each document using a simple gensim pre-processing tool (i.e., tokenize text into individual words, remove punctuation, set to lowercase, etc), and return a list of words. Also, to train the model, we'll need to associate a tag/number with each document of the training corpus. In our case, the tag is simply the zero-based line number.

# In[50]:

def read_corpus(documents):
    for i, plot in enumerate(documents):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(plot, max_len=30), [i])


# In[51]:

train_corpus = list(read_corpus(dataframe.abstract))


# Let's take a look at the training corpus.

# In[52]:

train_corpus[:2]


# ## Training the Doc2Vec Model
# We'll instantiate a Doc2Vec model with a vector size with 50 words and iterating over the training corpus 55 times. We set the minimum word count to 2 in order to give higher frequency words more weighting. Model accuracy can be improved by increasing the number of iterations but this generally increases the training time. Small datasets with short documents, like this one, can benefit from more training passes.

# In[53]:

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)


# Now, we'll save the document embedding vectors per doctag.

# In[54]:

model.save_word2vec_format('doc_tensor.w2v', doctag_vec=True, word_vec=False)  


# ## Prepare the Input files for Tensorboard

# Tensorboard takes two Input files. One containing the embedding vectors and the other containing relevant metadata. We'll use a gensim script to directly convert the embedding file saved in word2vec format above to the tsv format required in Tensorboard.

# In[55]:

get_ipython().magic('run ../../gensim/scripts/word2vec2tensor.py -i doc_tensor.w2v -o abstract_plot')


# The script above generates two files, `movie_plot_tensor.tsv` which contain the embedding vectors and `movie_plot_metadata.tsv`  containing doctags. But, these doctags are simply the unique index values and hence are not really useful to interpret what the document was while visualizing. So, we will overwrite `movie_plot_metadata.tsv` to have a custom metadata file with two columns. The first column will be for the movie titles and the second for their corresponding genres.

# In[57]:

with open('movie_plot_metadata.tsv','w') as w:
    w.write('Titles\tGenres\n')
    for i,j in zip(dataframe.title, dataframe.label):
        w.write("%s\t%s\n" % (i,j))


# Now you can go to http://projector.tensorflow.org/ and upload the two files by clicking on *Load data* in the left panel.
# 
# For demo purposes I have uploaded the Doc2Vec embeddings generated from the model trained above [here](https://github.com/parulsethi/DocViz). You can access the Embedding projector configured with these uploaded embeddings at this [link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/DocViz/master/movie_plot_config.json).

# # Using Tensorboard

# For the visualization purpose, the multi-dimensional embeddings that we get from the Doc2Vec model above, needs to be  downsized to 2 or 3 dimensions. So that we basically end up with a new 2d or 3d embedding which tries to preserve information from the original multi-dimensional embedding. As these vectors are reduced to a much smaller dimension, the exact cosine/euclidean distances between them are not preserved, but rather relative, and hence as you’ll see below the nearest similarity results may change.
# 
# TensorBoard has two popular dimensionality reduction methods for visualizing the embeddings and also provides a custom method based on text searches:
# 
# - **Principal Component Analysis**: PCA aims at exploring the global structure in data, and could end up losing the local similarities between neighbours. It maximizes the total variance in the lower dimensional subspace and hence, often preserves the larger pairwise distances better than the smaller ones. See an intuition behind it in this nicely explained [answer](https://stats.stackexchange.com/questions/176672/what-is-meant-by-pca-preserving-only-large-pairwise-distances) on stackexchange.
# 
# 
# - **T-SNE**: The idea of T-SNE is to place the local neighbours close to each other, and almost completely ignoring the global structure. It is useful for exploring local neighborhoods and finding local clusters. But the global trends are not represented accurately and the separation between different groups is often not preserved (see the t-sne plots of our data below which testify the same).
# 
# 
# - **Custom Projections**: This is a custom bethod based on the text searches you define for different directions. It could be useful for finding meaningful directions in the vector space, for example, female to male, currency to country etc.
# 
# You can refer to this [doc](https://www.tensorflow.org/get_started/embedding_viz) for instructions on how to use and navigate through different panels available in TensorBoard.

# ## Visualize using PCA
# 
# The Embedding Projector computes the top 10 principal components. The menu at the left panel lets you project those components onto any combination of two or three. 
# <img src="pca.png">
# The above plot was made using the first two principal components with total variance covered being 36.5%.

# 
# ## Visualize using T-SNE
# 
# Data is visualized by animating through every iteration of the t-sne algorithm. The t-sne menu at the left lets you adjust the value of it's two hyperparameters. The first one is **Perplexity**, which is basically a measure of information. It may be viewed as a knob that sets the number of effective nearest neighbors<sup>[2]</sup>. The second one is **learning rate** that defines how quickly an algorithm learns on encountering new examples/data points.
# 
# <img src="tsne.png">
# 
# The above plot was generated with perplexity 8, learning rate 10 and iteration 500. Though the results could vary on successive runs, and you may not get the exact plot as above with same hyperparameter settings. But some small clusters will start forming as above, with different orientations.

# # 2. Visualizing LDA
# 
# In this part, we will see how to visualize LDA in Tensorboard. We will be using the Document-topic distribution as the embedding vector of a document. Basically, we treat topics as the dimensions and the value in each dimension represents the topic proportion of that topic in the document.
# 
# ## Preprocess Text
# 
# We use the journal abstract as our documents in corpus and remove rare words and common words based on their document frequency. Below we remove words that appear in less than 2 documents or in more than 30% of the documents.

# In[61]:

import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
from gensim.models import ldamodel
from gensim.corpora.dictionary import Dictionary

# read data
# /Users/Shemelis/10-16-2017-GW-Arlington-Class-Repository-DATA/Machine-Learning-Project/All_Journal_data.csv

dataframe = pd.read_csv('/Users/Shemelis/10-16-2017-GW-Arlington-Class-Repository-DATA/Machine-Learning-Project/All_Journal_data.csv')

# remove stopwords and punctuations
def preprocess(row):
    return strip_punctuation(remove_stopwords(row.lower()))
    
dataframe['Plots'] = dataframe['abstract'].apply(preprocess)

# Convert data to required input format by LDA
texts = []
for line in dataframe.abstract:
    lowered = line.lower()
    words = re.findall(r'\w+', lowered, flags = re.UNICODE )
    texts.append(words)
# Create a dictionary representation of the documents.
dictionary = Dictionary(texts)

# Filter out words that occur less than 2 documents, or more than 30% of the documents.
dictionary.filter_extremes(no_below=2, no_above=0.3)
# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(text) for text in texts]


# ## Train LDA Model
# 

# In[65]:

# Set training parameters.
num_topics = 6
chunksize = 2000
passes = 5 #another name of epoch
iterations = 20
eval_every = None

# Train model
model = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every)


# You can refer to [this notebook](lda_training_tips.ipynb) also before training the LDA model. It contains tips and suggestions for pre-processing the text data, and how to train the LDA model to get good results.

# ## Doc-Topic distribution
# 
# Now we will use `get_document_topics` which infers the topic distribution of a document. It basically returns a list of (topic_id, topic_probability) for each document in the input corpus.

# In[66]:

# Get document topics
all_topics = model.get_document_topics(corpus, minimum_probability=0)
all_topics[0]


# The above output shows the topic distribution of first document in the corpus as a list of (topic_id, topic_probability).
# 
# Now, using the topic distribution of a document as it's vector embedding, we will plot all the documents in our corpus using Tensorboard.

# ## Prepare the Input files for Tensorboard
# 
# Tensorboard takes two input files, one containing the embedding vectors and the other containing relevant metadata. As described above we will use the topic distribution of documents as their embedding vector. Metadata file will consist of Movie titles with their genres.

# In[69]:

# create file for tensors
with open('doc_lda_tensor.tsv','w') as w:
    for doc_topics in all_topics:
        for topics in doc_topics:
            w.write(str(topics[1])+ "\t")
        w.write("\n")
        
# create file for metadata
with open('doc_lda_metadata.tsv','w') as w:
    w.write('Titles\ttitle\n')
    for j, k in zip(dataframe.title, dataframe.abstract):
        w.write("%s\t%s\n" % (j, k))


# Now you can go to http://projector.tensorflow.org/ and upload these two files by clicking on Load data in the left panel.
# 
# For demo purposes I have uploaded the LDA doc-topic embeddings generated from the model trained above [here](https://github.com/parulsethi/LdaProjector/). You can also access the Embedding projector configured with these uploaded embeddings at this [link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/LdaProjector/master/doc_lda_config.json).

# ## Visualize using PCA
# 
# The Embedding Projector computes the top 10 principal components. The menu at the left panel lets you project those components onto any combination of two or three.
# <img src="doc_lda_pca.png">
# From PCA, we get a simplex (tetrahedron in this case) where each data point represent a document. These data points are  colored according to their Genres which were given in the Movie dataset. 
# 
# As we can see there are a lot of points which cluster at the corners of the simplex. This is primarily due to the sparsity of vectors we are using. The documents at the corners primarily belongs to a single topic (hence, large weight in a single dimension and other dimensions have approximately zero weight.) You can modify the metadata file as explained below to see the dimension weights along with the Movie title.
# 
# Now, we will append the topics with highest probability (topic_id, topic_probability) to the document's title, in order to explore what topics do the cluster corners or edges dominantly belong to. For this, we just need to overwrite the metadata file as below:

# In[70]:

tensors = []
for doc_topics in all_topics:
    doc_tensor = []
    for topic in doc_topics:
        if round(topic[1], 3) > 0:
            doc_tensor.append((topic[0], float(round(topic[1], 3))))
    # sort topics according to highest probabilities
    doc_tensor = sorted(doc_tensor, key=lambda x: x[1], reverse=True)
    # store vectors to add in metadata file
    tensors.append(doc_tensor[:5])

# overwrite metadata file
i=0
with open('doc_lda_metadata.tsv','w') as w:
    w.write('Titles\tGenres\n')
    for j,k in zip(dataframe.title, dataframe.label):
        w.write("%s\t%s\n" % (''.join((str(j), str(tensors[i]))),k))
        i+=1


# Next, we upload the previous tensor file "doc_lda_tensor.tsv" and this new metadata file to http://projector.tensorflow.org/ .
# <img src="topic_with_coordinate.png">
# Voila! Now we can click on any point to see it's top topics with their probabilty in that document, along with the title. As we can see in the above example, "Beverly hill cops" primarily belongs to the 0th and 1st topic as they have the highest probability amongst all.
# 
# 
# 
# ## Visualize using T-SNE
# 
# In T-SNE, the data is visualized by animating through every iteration of the t-sne algorithm. The t-sne menu at the left lets you adjust the value of it's two hyperparameters. The first one is Perplexity, which is basically a measure of information. It may be viewed as a knob that sets the number of effective nearest neighbors[2]. The second one is learning rate that defines how quickly an algorithm learns on encountering new examples/data points.
# 
# Now, as the topic distribution of a document is used as it’s embedding vector, t-sne ends up forming clusters of documents belonging to same topics. In order to understand and interpret about the theme of those topics, we can use `show_topic()` to explore the terms that the topics consisted of.
# 
# <img src="doc_lda_tsne.png">
# 
# The above plot was generated with perplexity 11, learning rate 10 and iteration 1100. Though the results could vary on successive runs, and you may not get the exact plot as above even with same hyperparameter settings. But some small clusters will start forming as above, with different orientations.
# 
# I named some clusters above based on the genre of it's movies and also using the `show_topic()` to see relevant terms of the topic which was most prevelant in a cluster. Most of the clusters had doocumets belonging dominantly to a single topic. For ex. The cluster with movies belonging primarily to topic 0 could be named Fantasy/Romance based on terms displayed below for topic 0. You can play with the visualization yourself on this [link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/LdaProjector/master/doc_lda_config.json) and try to conclude a label for clusters based on movies it has and their dominant topic. You can see the top 5 topics of every point by hovering over it.
# 
# Now, we can notice that their are more than 10 clusters in the above image, whereas we trained our model for `num_topics=10`. It's because their are few clusters, which has documents belonging to more than one topic with an approximately close topic probability values.

# In[71]:

model.show_topic(topicid=0, topn=15)


# You can even use pyLDAvis to deduce topics more efficiently. It provides a deeper inspection of the terms highly associated with each individual topic. For this, it uses a measure called **relevance** of a term to a topic that allows users to flexibly rank terms best suited for a meaningful topic interpretation. It's weight parameter called λ can be adjusted to display useful terms which could help in differentiating topics efficiently.

# In[73]:

import pyLDAvis.gensim

viz = pyLDAvis.gensim.prepare(model, corpus, dictionary)
pyLDAvis.display(viz)


# The weight parameter λ can be viewed as a knob to adjust the ranks of the terms based on whether they are simply ranked according to their probability in the topic (λ=1) or are normalized by their marginal probability across the corpus (λ=0). Setting λ=1 could result in similar ranking of terms for large no. of topics hence making it difficult to differentiate between them, and setting λ=0 ranks terms solely based on their exclusiveness to current topic which could result in such rare terms that occur in only a single topic and hence the topics may remain difficult to interpret. [(Sievert and Shirley 2014)](https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf) suggested the optimal value of λ=0.6 based on a user study.

# # Conclusion
# 
# We learned about visualizing the Document Embeddings and LDA Doc-topic distributions through Tensorboard's Embedding Projector. It is a useful tool for visualizing different types of data for example, word embeddings, document embeddings or the gene expressions and biological sequences. It just needs an input of 2D tensors and then you can explore your data using provided algorithms. You can also perform nearest neighbours search to find most similar data points to your query point.
# 
# # References
#  1. https://grouplens.org/datasets/movielens/
#  2. https://lvdmaaten.github.io/tsne/
# 
