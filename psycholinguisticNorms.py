import numpy as np
import spacy
import pickle
from glob import iglob as glob
import os.path
from os.path import basename
from tqdm import *
import argparse

################################################################
# I use spacy as my pre-processor but you migth want to use NLTK
################################################################
nlp = spacy.load('en')

################################################################
# Aux function to obtain functionals
################################################################
funcs = [np.mean, np.max, np.min, np.std]
funcs_names = ["mean", "max", "min", "std"]

def functionals(vals):
    return [f(vals) for f in funcs]

################################################################
#
################################################################
# parser = argparse.ArgumentParser(description='Calculate emotiword norms')
# parser.add_argument('inputdir', help = "Input directory")
# parser.add_argument('outdir', help = "Output directory")
# parser.add_argument('--normsdir', default = ".", help = "Input file")
# parser.add_argument('--debug', default = False, action='store_false', help = "debug")

# args = parser.parse_args()

################################################################
# debug
################################################################
# if args.debug:
#     print("Input dir: {}".format(args.inputdir))
#     print("Output dir: {}".format(args.outdir))
#     print("Emotiword dir: {}".format(args.normsdir))

################################################################
# Load Emotiword
################################################################

emotiword = {}
for dim in glob(os.path.join('./Emotiword', "*.pickle")):
    with open(dim, 'rb') as f:
        emotiword[basename(dim).replace(".pickle", "")] = pickle.load(f, encoding = 'latin1')

################################################################
# These are the tags I'm interest in. I'll calculate functionals
# only on these.
# These depend on the preprocessor so it might be different for
# NLTK.
################################################################
keep_pos = sorted(["ADJ", "ADV", "VERB"])

################################################################
# Process text
# Returns a tensor of shape
#        #tags x #emotiword dimensions x #functionals
################################################################
def processText(text):
    norms_ratings = np.zeros((len(keep_pos), len(emotiword), len(funcs)))
    pos_dict = {x:[] for x in keep_pos}


    ################################################################
    # Spacy gives me tokenizer and POS
    ################################################################
    for word in nlp(text):

        if word.pos_ not in keep_pos:
            continue

        if word.orth_ in emotiword['aro']:
            vector = [emotiword[dim][word.orth_] for dim in emotiword]

            pos_dict[word.pos_].append(vector)

            #########################################################################
            # This is a hack to calculate tags V and J, which do not appear in spacy
            #########################################################################
            # pos_dict[word.tag_[0]].append(vector)

    ################################################################
    # Calculate functionals of every key
    ################################################################
    for i, k in enumerate(keep_pos):
        if len(pos_dict[k]) > 0:
            t = np.array(pos_dict[k]).T #dim_in_emotiword x #words_in_emotiword
            for j in range(len(emotiword)):
                norms_ratings[i, j, :] = functionals(t[j])

    return norms_ratings


# def processText_toDict(text):
#     norms_ratings = {}
#     pos_dict = {x:[] for x in keep_tag}

#     for word in nlp(text):

#         if word.tag_ not in keep_tag:
#             continue

#         if word.orth_ in emotiword['aro']:
#             vector = {dim : emotiword[dim][word.orth_] for dim in emotiword}

#             pos_dict[word.tag_].append(vector)

#             pos_dict[word.tag_[0]].append(vector)

#     for i, k in enumerate(keep_tag):
#         if len(pos_dict[k]) > 0:
#             norms_ratings[k] = {dim : {name : f(pos_dict[k]) for name, f in zip(funcs_names, funcs)} for dim in emotiword}



################################################################
# Directory mode
################################################################
# if args.inputdir is not None:
#     for f in tqdm(list(glob(args.inputdir))):
#         with open(f) as inpt:
#             lines = inpt.readlines()
#             ratings = np.empty((len(lines), len(keep_pos), len(emotiword), len(funcs)))
#
#             for idx, line in enumerate(lines):
#                 ratings[idx, :] = processText(line)
#
#         ################################################################
#         # Save normative tensor to a file
#         ################################################################
#         np.save(os.path.join(args.outdir, '{}_ratings.nparray'.format(basename(f).replace(".txt", ""))), ratings)
