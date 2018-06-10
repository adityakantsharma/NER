#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
import re

from convert_conll2spacy import convert_conll2spacy

# training data
TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'I-PER')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'I-LOC'), (18, 24, 'I-LOC')]
    })
]
    
TEST_DATA = ('Where is Aditya ?',)



@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=5):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    traindata = 'data/conll03/eng.train'
    testdata = 'data/conll03/eng.testa' 
    Cv = convert_conll2spacy(traindata)
    train_data = Cv.convert()[0]
    Cv = convert_conll2spacy(testdata)
    test_data = Cv.convert()[0]
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

#    # test the trained model
    tagged_output = 'spacy_trained_with_conll_tested_on_conlltesta'
#    #testfilespacygermeval = open(tagged_output, "w")
#    for word in doc:
#        #print(word.text, word.orth, word.lower, word.tag_, word.ent_type_, word.ent_iob)
#        line = word.text + "\t" + word.ent_type_ + "\n"
#        testfilespacygermeval.write(line)
#        i += 1
#    print(i)    
    
#    with open(testdata, "r") as test_file:
#        lines = test_file.readlines()
#        with open(tagged_output, "w") as testfilespacygermeval:
#            for line in lines:
#                text = re.split(" ", line)[0]
#                doc = nlp(text)
#                print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
#                print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
#                for word in doc:
#                    line = word.text + "\t" + word.ent_type_ + "\n"
#                    testfilespacygermeval.write(line)
#                    postprocess(tagged_output)


    with open(tagged_output, "w") as testfilespacygermeval:
        for text, _ in test_data:
            #text = re.split(" ", line)[0]
            doc = nlp(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
            for word in doc:
                line = word.text + "\t" + word.ent_type_ + "\n"
                testfilespacygermeval.write(line)
                postprocess(tagged_output)


    # save model to output directory
    output_dir = 'classifiers/spacy/spacy_conll_trained'
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
#        print("Loading from", output_dir)
#        nlp2 = spacy.load(output_dir)
#        for text, _ in train_data:
#            doc = nlp2(text)
#            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
#            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

def postprocess(tagged_files):
    i = 1
    output = tagged_files + ".tsv"
    with open(tagged_files, "r") as pre, open(output, "w") as post:
        for line in pre:
            line = line.split("\t")
            # Fixing ignored tokens in germaner conll formated files by stanford ner on lines 64899 and 99279
            #if i == 64899 or i == 99279:
            #    post.write("<>" + "\t" + "O" + "\n")
            if len(line) >= 1:
                if line[0] == "####":
                    post.write("")
                elif line[0] == " ":
                    post.write("\n")
                else:
                    post.write(line[0] + "\t" + line[1])
            else:
                print(line, i)
            i += 1

if __name__ == '__main__':
    plac.call(main)

    # Expected output:
    # Entities [('Shaka Khan', 'PERSON')]
    # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
    # ('Khan', 'PERSON', 1), ('?', '', 2)]
    # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
    # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3),
    # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]