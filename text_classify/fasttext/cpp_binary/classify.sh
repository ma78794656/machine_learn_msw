cur_dir=$(pwd)
# build fasttext c++ binary
cd fastText && make clean && make && cd ..
# parse data
python=/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/bin/python3.6
${python} parse_data.py
# train and test model
train_file=train.csv
output_name=model
output_file=model.bin
test_file=test.csv
./fastText/fasttext supervised -input $train_file -output $output_name -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4

./fastText/fasttext test $output_file $test_file

./fastText/fasttext predict $output_file $test_file > "${test_file}.predict"
