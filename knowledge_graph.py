import nltk
import sys
import pickle
import os
from collections import defaultdict
import glob
# For Spacy:
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
import pandas as pd
# For custom ER:
import tkinter
import re
# For Coreference resolution
import json
from stanfordcorenlp import StanfordCoreNLP

class StanfordNER:
    def __init__(self):
        self.get_stanford_ner_location()

    def get_stanford_ner_location(self):  
        loc = "/content/knowledge_graph_from_unstructured_text/stanford-ner-2018-10-16"
        print("... Running stanford for NER; this may take some time ...")        
        self.stanford_ner_tagger = nltk.tag.StanfordNERTagger(loc+'/classifiers/english.all.3class.distsim.crf.ser.gz',
        loc+'/stanford-ner.jar')

    def ner(self,doc):
        sentences = nltk.sent_tokenize(doc)
        result = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            tagged = self.stanford_ner_tagger.tag(words)
            result.append(tagged)
        return result

    def display(self,ner):
        print(ner)
        print("\n")

    def ner_to_dict(self,ner):
        """
        Expects ner of the form list of tuples
        """
        ner_dict = {}
        for tup in ner:
            ner_dict[tup[0]] = tup[1]
        return ner_dict

class CoreferenceResolver:
    def generate_coreferences(self,doc,stanford_core_nlp_path,verbose):
        '''
        pickles results object to coref_res.pickle
        the result has the following structure:
        dict of dict of lists of dicts:  { { [ {} ] } }  -- We are interested in the 'corefs' key { [ { } ] }-- Each list has all coreferences to a given pronoun.
        '''
        nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet =  not verbose)
        props = {'annotators': 'coref', 'pipelineLanguage': 'en'}
        annotated = nlp.annotate(doc, properties=props)
        print("\nannotated\n\n", annotated, "\n\n")
        result = json.loads(annotated)
        # Dump coreferences to a file
        pickle.dump(result,open( "coref_res.pickle", "wb" ))
        # Close server to release memory
        nlp.close()
        return result

    def display_dict(self,result):
        for key in result:
            print(key,":\n",result[key]) 
            print("\n")

    def unpickle(self):
        result = pickle.load(open( "coref_res.pickle", "rb" ))
        return result
    
    def resolve_coreferences(self,corefs,doc,ner,verbose):
        """
        Changes doc's coreferences to match the entity present in ner provided.
        ner must be a dict with entities as keys and names/types as values
        E.g. { "Varun" : "Person" }
        """
        corefs = corefs['corefs']
        if verbose:
            print("Coreferences found: ",len(corefs),"\nThe coreferences are:")
            self.display_dict(corefs)
            print("Named entities:")
            print(ner.keys())

        # replace all corefs in i th coref list with this
        replace_coref_with = []
        
        # Key is sentence number; value is list of tuples. 
        # Each tuple is (reference_dict, coreference number)
        sentence_wise_replacements = defaultdict(list)         # { 0: [ ({},ref#),({},ref#), ...], 1: [({}) ...]... }  

        sentences = nltk.sent_tokenize(doc)
        for index,coreferences in enumerate(corefs.values()):    # corefs : {[{}]} => coreferences : [{}]
            # Find which coreference to replace each coreference with. By default, replace with first reference.
            replace_with = coreferences[0]
            for reference in coreferences:      # reference : {}
                if reference["text"] in ner.keys() or reference["text"][reference["headIndex"]-reference["startIndex"]] in ner.keys():
                    replace_with = reference
                sentence_wise_replacements[reference["sentNum"]-1].append((reference,index))
            replace_coref_with.append(replace_with["text"])  
        
        # sort tuples in list according to start indices for replacement 
        sentence_wise_replacements[0].sort(key=lambda tup: tup[0]["startIndex"]) 

        if verbose:
            for key,val in sentence_wise_replacements.items():
                print("Sent no# ",key)
                for item in val:
                    print(item[0]["text"]," ",item[0]["startIndex"]," ",item[0]["endIndex"]," -> ",replace_coref_with[item[1]]," replacement correl #",item[1], end ="   ||| ")
                print("\n")

        
        #Carry out replacement
        for index,sent in enumerate(sentences):
            # Get the replacements in ith sentence
            replacement_list = sentence_wise_replacements[index]    # replacement_list : [({},int)]
            # Replace from last to not mess up previous replacement's indices
            for item in replacement_list[::-1]:                     # item : ({},int)
                to_replace = item[0]                                # to_replace: {}
                replace_with = replace_coref_with[item[1]]
                replaced_sent = ""
                words = nltk.word_tokenize(sent)
                
                # replace only if what is inted to be replaced is the thing we are trying to replace
                # to_be_replaced = ""
                # for i in range(to_replace["startIndex"],to_replace["endIndex"]):
                #     to_be_replaced  += words[i]
                # if verbose:
                #     print("Intended Replacement: ", to_replace["text"])
                #     print("What's to be replaced: ", to_be_replaced)
                # if to_be_replaced != to_replace["text"]:
                #     if verbose:
                #         print("Texts do not match, skipping replacement")
                #     continue

                if verbose:
                    print("Original: ",sent)
                    print("To replace:", to_replace["text"]," | at:",to_replace["startIndex"],to_replace["endIndex"],end='')
                    print(" With: ",replace_with)
                # Add words from end till the word(s) that need(s) to be replaced
                for i in range(len(words)-1,to_replace["endIndex"]-2,-1):
                    replaced_sent = words[i] + " "+ replaced_sent
                # Replace
                replaced_sent = replace_with + " " + replaced_sent
                # Copy starting sentence
                for i in range(to_replace["startIndex"]-2,-1,-1):
                    replaced_sent = words[i] + " "+ replaced_sent
                if verbose:
                    print("Result: ",replaced_sent,"\n\n")
                sentences[index] = replaced_sent

        result = ""
        for sent in sentences:
            result += sent
        if verbose:
            print("Original text: \n",doc)
            print("Resolved text:\n ",result)
        return result

def resolve_coreferences(doc,stanford_core_nlp_path,ner,verbose):
    coref_obj = CoreferenceResolver()
    corefs = coref_obj.generate_coreferences(doc,stanford_core_nlp_path,verbose)
    #coref.unpickle()
    result = coref_obj.resolve_coreferences(corefs,doc,ner,verbose)
    return result

def main():

    output_path = "/content/knowledge_graph_from_unstructured_text/data/output/"
    ner_pickles_op = output_path + "ner/"
    coref_cache_path = output_path + "caches/"
    coref_resolved_op = output_path + "kg/"
    
    stanford_core_nlp_path = "/content/stanford-nlp/stanford-corenlp-4.2.2"
    file_list = []
    for f in glob.glob('/content/knowledge_graph_from_unstructured_text/data/input/*'):
        file_list.append(f)
 
    for file in file_list:
        with open(file,"r") as f:
            lines = f.read().splitlines()
        
        doc = ""
        for line in lines:
            doc += line


        print("using Stanford for NER (may take a while):  \n\n\n")
        stanford_ner = StanfordNER()
        tagged = stanford_ner.ner(doc)
        ner = stanford_ner.ner(doc)
        stanford_ner.display(ner)

        # ToDo -- Implement ner_to_dict for stanford_ner
        named_entities = stanford_ner.ner_to_dict(stanford_ner.ner(doc))


        # Save named entities
        op_pickle_filename = ner_pickles_op + "named_entity_"+file.split('/')[-1].split('.')[0]+".pickle"
        with open(op_pickle_filename,"wb") as f:
            pickle.dump(named_entities, f)

main()
