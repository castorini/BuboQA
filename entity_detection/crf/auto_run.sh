
# Download the software from the website (version 2017-06-09)
if [ ! -f stanford-ner.zip ]; then
    wget -c -t 0 -O stanford-ner.zip https://nlp.stanford.edu/software/stanford-ner-2017-06-09.zip
fi
if [ ! -d stanford-ner ]; then
    unzip stanford-ner.zip
    mv stanford-ner-2017-06-09 stanford-ner
fi
pushd stanford-ner
pwd
mkdir data
## Convert
python ../convert.py

echo "Training in domain data"
java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop ../qa.prop

echo "Testing"
java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier qa_ner.ser.gz -testFile data/stanford.valid > data/stanford.predicted.valid
java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier qa_ner.ser.gz -testFile data/stanford.test > data/stanford.predicted.test
echo "Evaluation on test data (in-domain)"
python ../eval.py data/stanford.predicted.valid
python ../eval.py data/stanford.predicted.test

popd

mkdir query_text
python output2query.py
