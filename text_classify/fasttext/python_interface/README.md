1. The python version is 3.6.2, may or may not run in other versions.
2. Use facebook fasttext tool and 'meituan waimai dingdan' to train the model.
3. Parameters are not optimized(about 85% P and R)
4. Avaliable params and their default value:
   input_file               training file path (required)
   output                   output file path (required)
   label_prefix             label prefix ['__label__']
   lr                       learning rate [0.1]
   lr_update_rate           change the rate of updates for the learning rate [100]
   dim                      size of word vectors [100]
   ws                       size of the context window [5]
   epoch                    number of epochs [5]
   min_count                minimal number of word occurences [1]
   neg                      number of negatives sampled [5]
   word_ngrams              max length of word ngram [1]
   loss                     loss function {ns, hs, softmax} [softmax]
   bucket                   number of buckets [0]
   minn                     min length of char ngram [0]
   maxn                     max length of char ngram [0]
   thread                   number of threads [12]
   t                        sampling threshold [0.0001]
   silent                   disable the log output from the C++ extension [1]
   encoding                 specify input_file encoding [utf-8]
   pretrained_vectorspretrained word vectors (.vec file) for supervised learning []
5. Ref git repository: https://github.com/ma78794656/fastText.py
