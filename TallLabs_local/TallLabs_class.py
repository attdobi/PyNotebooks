#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division,unicode_literals
import os
base_dir=os.path.expanduser('~') #get home dir
import pandas as pd
import datetime
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
import re, collections, random, datetime
import psycopg2
import json
from gensim import corpora, models, similarities
from sklearn.externals import joblib

# guarantee unicode string... #no need in python3 (will cause an error)
_u = lambda t: t.decode('UTF-8', 'replace') if isinstance(t, str) else t

class TallLabs_lib:
	"""Tall Labs Class"""
	def __init__(self):
		#setup
		self.conn = psycopg2.connect("host=localhost port=5432 dbname=amazon user=postgres password=darkmatter")
		self.cur = self.conn.cursor()
		self.stoplist = set('a an for of the and to in rt'.split())
		self.clf = joblib.load(base_dir+'/TallLabs/models/three_word_logreg_py2.pkl') 
		self.clf_1 = joblib.load(base_dir+'/TallLabs/models/first_word_logreg_py.pkl') 
		self.QmodelB=models.Word2Vec.load(base_dir+'/TallLabs/models/QmodelB')
		self.RmodelB=models.Word2Vec.load(base_dir+'/TallLabs/models/RmodelB_cell')
		self.Rmodel_D2V=models.Doc2Vec.load('/home/ubuntu/TallLabs/models/Rmodel_Doc2vec_cell')
		self.lda=models.LdaModel.load(base_dir+'/TallLabs/models/lda_cell_15')
		self.dictionary=corpora.Dictionary.load(base_dir+'/TallLabs/models/lda_cell_dict_15')
		self.bag_of_words_yn='is,will,wil,may,might,does,dose,doe,dos,do,can,could,must,should,are,would,do,did'.split(',')
		self.bag_of_words_oe="what,what's,where".split(',')
		self.bag_of_words='is,will,wil,may,might,does,do,can,could,must,should,are,would,did,need,take,out,how,would,am,at,\
anyone,has,have,off,that,which,who,please,thank,you,that,fit,these,they,many,work,with,time,turn,fit,fitt,going,\
from,hard,use,your,not,into,non,hold,say,from,one,two,like,than,same,thanks,find,make,hot,be,as,well,there,\
son,daughter,amazon,when,after,change,both,ask,know,help,me,recently,purchased,item,any,newest,or,come,hi'.split(',')
		self.bag_of_words_verbs='is,will,wil,may,might,does,do,can,could,must,should,are,would,did,take,out,would,\
anyone,off,that,which,who,please,thank,you,that,these,they,many,time,turn,newest,there,am,at,\
from,hard,use,your,not,into,non,hold,say,from,one,two,like,than,same,thanks,\
son,daughter,amazon,when,after,change,both,ask,know,help,me,recently,purchased,item,any,hi'.split(',')
		self.complete_bag=set(sum([[item[0] for item in self.QmodelB.most_similar(word)] for word in self.bag_of_words],[]))|self.stoplist|set(self.bag_of_words)
		self.complete_bag_verbs=set(sum([[item[0] for item in self.QmodelB.most_similar(word)] for word in self.bag_of_words_verbs],[]))|self.stoplist|set(self.bag_of_words_verbs)
		#LDA categories
		self.LDAcategories={0:'Clips, Mounts, Holsters ',1:'Cables, Chargers, Adapters',2:'Batteries, Battery Life',3:'Product Description',\
		4:'USB, Ports, Power',5:'Protective Covers',6:'Prices, Quality',7:'Product Size',\
		8:'Car Accessories, GPS',9:'Screen Protector',10:'Refunds',11:'Bluetooth, Headsets',12:'WaterProof',\
		13:'Camera,Apps',14:'Brands, Models'}
		
	def clean_result(self,model_result):
		return [item[0] for item in model_result],[item[1] for item in model_result]
		#topn=15
	def clean_result_Doc(self,most_sim):
		similar_asins=[val[0].split('R_')[1] for val in most_sim]
		similarity=[val[1] for val in most_sim]
		return similar_asins,similarity
		
	def return_title(self,asin):
		self.cur.execute("SELECT title FROM metadata_cell_phones_and_accessories WHERE asin=%s limit 1;",(asin,))
		return ' '.join(self.cur.fetchall()[0][0].split()[:5])
		
	def visual(self,word,model):
		if model=='reviews':
			modelB=self.RmodelB
		elif model=='questions':
			modelB=self.QmodelB
		else:
			modelB=self.QmodelB
			
		word=word.lower()
		results,counts=self.clean_result(modelB.most_similar(word,topn=5))
		source_target=[]
		for result_word in results:
			#append source target, and search next layer
			source_target.append((word,result_word,1))
			results2,counts2=self.clean_result(modelB.most_similar(result_word,topn=5))
			for result_word2 in results2:
				source_target.append((result_word,result_word2,2))
				results3,counts3=self.clean_result(modelB.most_similar(result_word2,topn=5))
				for result_word3 in results3:
					source_target.append((result_word2,result_word3,3))
		return [{"source":src,"target":tar,"group":grp} for src,tar,grp in source_target]
		
	def tree(self,word,model):
		if model=='reviews':
			modelB=self.RmodelB
		elif model=='questions':
			modelB=self.QmodelB
		else:
			modelB=self.QmodelB
			
		word=word.lower()
		results,counts=self.clean_result(modelB.most_similar(word))
		target=[]
		child_list=[]
		for result_word in results:
			target_child=[]
			target.append(result_word)
			results2,counts2=self.clean_result(modelB.most_similar(result_word))
			for result_word2 in results2:
				target_child.append(result_word2)
			child_list.append(target_child)
		return {"name":word,"children":[{"name":tar,"children":[{"name":child,"size":3} for child in child_l] }\
		 for tar,child_l in zip(target,child_list)]}
		 
	def tree_Doc(self,head_asin,keys):
		modelDoc2vec=self.Rmodel_D2V
		key_words=keys.lower().replace(', ', ',').replace(' ,', ',').replace(' , ', ',').split(',')# form into an array
		key_words_bigram=[word.replace(' ','_') for word in key_words]
		similar_keys=sum([[' '.join(item[0].split('_')) for item in self.check_key(word,'review') if item!=[''] and item[1]>0.7] for word in key_words_bigram],[])
		search_key_vector=modelDoc2vec.infer_vector(key_words+similar_keys,alpha=0) #set alpha to 0 to prevent random permutation
		key='R_'+head_asin
		similar_asins,similarity=self.clean_result_Doc(modelDoc2vec.docvecs.most_similar([key,7*search_key_vector]))
		target=[]
		child_list=[]
		for asin in similar_asins[:5]:
			target_child=[]
			target.append(asin+' , '+self.return_title(asin))
			similar_asins2,similarity2=self.clean_result_Doc(modelDoc2vec.docvecs.most_similar(['R_'+asin,7*search_key_vector]))
			for asin2 in similar_asins2[:5]:
				target_child.append(asin2+' , '+self.return_title(asin2))
			child_list.append(target_child)
		return {"name":head_asin+' , '+self.return_title(head_asin),"children":[{"name":tar,"children":[{"name":child,"size":3} for child in child_l] }\
		 for tar,child_l in zip(target,child_list)]}
		 
	def train(self,input,name):
		self.cur.execute("SELECT qa_id from training order by id DESC limit 1;")
		last_id=self.cur.fetchall()
		self.cur.execute("SELECT id,question,questiontype from qa where id>%s limit 2;",last_id)
		result=self.cur.fetchall()
		qa_id=result[0][0]
		sentence=result[0][1] #returns sentence as a string (could be multiple from one query)
		sentence,first_word = self.process_line_question(sentence) #returns the first sentence and first word
		sentence2=result[1][1] #display this one if the current one is processed
		sentence2,first_word2 = self.process_line_question(sentence2)
		qestion_type=result[0][2]
		
		if input != 'start':
			self.cur.execute("SELECT data_corr_yn,data_corr_oe,bow_corr_yn,bow_corr_oe,count_yn,count_oe from training order by id DESC limit 1;")
			result=self.cur.fetchall()[0]
			data_corr_yn=result[0]
			data_corr_oe=result[1]
			bow_corr_yn=result[2]
			bow_corr_oe=result[3]
			count_yn=result[4]
			count_oe=result[5]
			#check if sentence is in bag of words
			print(self.first_word_in_bag(first_word))
			if self.first_word_in_bag(first_word):
				qestion_type_bow='yes/no'
			else:
				qestion_type_bow='open-ended'
			
			
			if (input == 'yes/no') & (qestion_type_bow =='yes/no'):
				bow_corr_yn= (bow_corr_yn*count_yn+1)/(count_yn+1)
			if (input == 'yes/no') & (qestion_type_bow !='yes/no'):
				bow_corr_yn= (bow_corr_yn*count_yn+0)/(count_yn+1)
			if (input == 'open-ended') & (qestion_type_bow =='open-ended'):
				bow_corr_oe= (bow_corr_oe*count_oe+1)/(count_oe+1)
			if (input == 'open-ended') & (qestion_type_bow !='open-ended'):
				bow_corr_oe= (bow_corr_oe*count_oe+0)/(count_oe+1)
				
			if (input == 'yes/no') & (qestion_type =='yes/no'):
				data_corr_yn= (data_corr_yn*count_yn+1)/(count_yn+1)
			if (input == 'yes/no') & (qestion_type !='yes/no'):
				data_corr_yn= (data_corr_yn*count_yn+0)/(count_yn+1)
			if (input == 'open-ended') & (qestion_type =='open-ended'):
				data_corr_oe= (data_corr_oe*count_oe+1)/(count_oe+1)
			if (input == 'open-ended') & (qestion_type !='open-ended'):
				data_corr_oe= (data_corr_oe*count_oe+0)/(count_oe+1)
				
			if (input == 'open-ended'):
				count_oe=count_oe+1
			if (input == 'yes/no'):
				count_yn=count_yn+1
			self.insert_result(sentence,qestion_type,qestion_type_bow,input,qa_id,data_corr_yn,data_corr_oe,bow_corr_yn,bow_corr_oe,count_yn,count_oe,name)
			sentence=sentence2 #return the next one to display
		return(sentence)
		 
	def train_plot(self):
		self.cur.execute("SELECT data_corr_yn,data_corr_oe,bow_corr_yn,bow_corr_oe from training order by id DESC limit 1;")
		result=self.cur.fetchall()[0]
		print(result)
		yn=[result[0],result[2]]
		oe=[result[1],result[3]]
		key=['Data','BoW']
		label=['Yes/No','Open Ended']
		#print(xx,yy,key)
		return [{"values":[{"y":yn[0]*100,"x":label[0]},{"y":oe[0]*100,"x":label[1]}],"key":key[0],"yAxis":"1"},\
 {"values":[{"y":yn[1]*100,"x":label[0]},{"y":oe[1]*100,"x":label[1]}],"key":key[1],"yAxis":"1"}]
	
	
	def leader_board(self):
		self.cur.execute("SELECT name, count(*) from training group by name order by count DESC;")
		result=self.cur.fetchall()
		#print(result)
		#names = [val[0] for val in result]
		#counts = [val[1] for val in result]
		return {"values":[{"rank":rank+1,"value":count,"label":name} for rank,(name,count) in enumerate(result)],"key": "Serie 1"}
		
	def insert_result(self,question,qestion_type,qestion_type_bow,qestion_type_human,qa_id,data_corr_yn,data_corr_oe,bow_corr_yn,bow_corr_oe,count_yn,count_oe,name):
		self.cur.execute("INSERT INTO training (\
		question,\
		qestion_type,\
		qestion_type_bow,\
		qestion_type_human,\
		qa_id,\
		data_corr_yn,\
		data_corr_oe,\
		bow_corr_yn,\
		bow_corr_oe,\
		count_yn,\
		count_oe,\
		name,\
		time\
		)\
		VALUES (\
		%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\
		)",(\
		question,\
		qestion_type,\
		qestion_type_bow,\
		qestion_type_human,\
		qa_id,\
		data_corr_yn,\
		data_corr_oe,\
		bow_corr_yn,\
		bow_corr_oe,\
		count_yn,\
		count_oe,\
		name,\
		datetime.datetime.now()\
		))
		self.conn.commit() #submit change to db
		
	def process_line(self,sentence):
		#step 1 split if we need to
		sentences=re.split(r'[;:.!?]\s*', sentence)
		result= [re.findall("[a-z-.'0-9]+", sent.lower()) for sent in sentences if \
				re.findall("[a-z-.'0-9]+", sent.lower())!=[]]
		if result==[]:
			result=[['']]
		return result
		
	def process_line_question(self,sentence):
		#step 1, split on question. Use for QA training page
		sentences=re.split(r'[;:.!?]\s*', sentence)
		result= [re.findall("[a-z-.'0-9]+", sent.lower()) for sent in sentences if \
			re.findall("[a-z-.'0-9]+", sent.lower())!=[]]
		#return first sentence and first word
		return ' '.join(result[0])+'?',result[0][0]
		
	def first_word_in_bag(self,first_word):
		try:
			is_in_bag=({first_word}|{item[0] for item in self.QmodelB.most_similar(first_word)})&set(self.bag_of_words_yn)!=set()
		except KeyError:
			is_in_bag=False
			
		#remove open ended question
		if first_word in self.bag_of_words_oe:
			is_in_bag=False
			
		return is_in_bag
		
	def getMeta(self,asin):
		self.cur.execute("select metajson->'imUrl', metajson->'description', title from metadata_cell_phones_and_accessories where asin=%s limit 1;",(asin,))
		result=self.cur.fetchall()[0]
		image=result[0]
		description=result[1]
		title=result[2]
		
		#get the question, only use for demo
		self.cur.execute("SELECT question from qa where asin=%s limit 1;",(asin,))
		result=self.cur.fetchall()[0]
		question=result[0]
		
		#get reviews
		self.cur.execute("select reviewtext from reviews_cell_phones_and_accessories where asin=%s;",(asin,))
		result=self.cur.fetchall()
		formated_reviews='\n\n'.join([val[0] for val in result])
		
		return image,title,description,question,formated_reviews
		
	def findTopic(self,key_words,similar_keys):
		'''Return the LDA vector composition of the topics based on the key words in the search'''
		key_words=[words.replace(' ','_') for words in key_words] #add back for bigram search
		similar_keys=[words.replace(' ','_') for words in similar_keys] #add back for bigram search
		#using the review model
		bow_vector = self.dictionary.doc2bow(key_words+similar_keys)
		lda_np=np.array(self.lda[bow_vector]) #find nearest lda vector topic
		lda_np=lda_np[lda_np[:,1].argsort()[::-1]] #sort by percentage 
		return '\n'.join(['{:.1f}%: '.format(val[1]*100)+self.LDAcategories[val[0]] for val in lda_np][:3])

	def processQuestion(self,asin,question):
		'''Find key words and return the most relevent answer from the review text.
		Using Doc2Vec find the most similar reviews and search for answers there also'''
		question.replace("- "," ").replace(" -"," ") #remove - in questions
		key_words, key_words_action = self.return_key_words(question)
		key_words_bigram=[word.replace(' ','_') for word in key_words]
		similar_keys=sum([[' '.join(item[0].split('_')) for item in self.check_key(word,'review') if item!=[''] and item[1]>0.7] for word in key_words_bigram],[])
		
		topic_text=self.findTopic(key_words,similar_keys)#find LDA topic based on search keys
		
		### pull review data on the asin of the product
		self.cur.execute("select reviewtext from reviews_cell_phones_and_accessories where asin=%s;",(asin,))
		result=self.cur.fetchall() #there are multiple reviews per asin, merge them next
		'''Next find the relavent lines in the review text sorted by number of keyword matches'''
		good_sen,good_qual,good_qual_val=self.find_relevent_sentence(self.merge_review(result),key_words)#+similar keys if nothing is found
		sorted_index=sorted(range(len(good_qual_val)),key=lambda x:good_qual_val[x])[::-1]
		#formatted_answer='\n\n'.join([good_qual[index]+':'+good_sen[index] for index in sorted_index][0:5])
		formatted_answer='\n\n'.join([str(ii+1)+':'+good_sen[index] for ii,index in enumerate(sorted_index)][0:5])
		
		'''Find similar reviews based on the nearest review document vecor and question keys(can cross check with amazon 'similar items' in meta data)'''
		similar_asins,sim_reviews,sim_images,sim_titles,sim_descriptions=self.similarReviews(asin,key_words+similar_keys,1)
		good_sen,good_qual,good_qual_val=self.find_relevent_sentence(sim_reviews[0],key_words)
		sorted_index=sorted(range(len(good_qual_val)),key=lambda x:good_qual_val[x])[::-1]
		formatted_answer_sim='\n\n'.join([str(ii+1)+':'+good_sen[index] for ii,index in enumerate(sorted_index)][0:5])
		
		'''get question type prediction based on logistic regresion model:'''
		words=re.findall("[a-z'0-9]+", question.lower())
		if len(words)>=3:
			try:
				prediction=self.clf.predict(1*self.QmodelB[words[0]]+self.QmodelB[words[1]]+self.QmodelB[words[2]])[0]
			except KeyError:
				preduction=1
		elif (len(words)<3) & (len(words)>0):
			try:
				prediction=self.clf_1.predict(self.QmodelB[words[0]])[0]
			except KeyError:
				preduction=1
		else:
			prediction=1
			
		if prediction==1:
			qType='Yes/No'
		else:
			qType='Open-Ended'
		###### End predict question type ##############################
		
		about_text='Question Type: '+qType + '\n\n'+\
		'Keywords: '+ ', '.join(key_words) + '\n'+\
		'Action Words: '+ ', '.join(key_words_action) + '\n'+\
		'Similar Keys: '+ ', '.join(similar_keys) +'\n\n'+\
		'SubTopics: \n'+ topic_text
		
		return formatted_answer, about_text, formatted_answer_sim,sim_titles[0],sim_images[0]
		
	###### Support functions for porcessQuetion ########################################################################
	def similarReviews(self,asin,key_words,N=1):
		'''Return asin and review text of the the N most reviews based on Doc2vec model
		search for most similar review which also includes Keys and Similar Keys from user's search'''
		model=self.Rmodel_D2V #Doc2Vec model trained on the cell phone and accessory review category
		search_key_vector=model.infer_vector(key_words,alpha=0) #set alpha to 0 to prevent random permutation
		most_sim=model.docvecs.most_similar(['R_'+asin,7*search_key_vector])[:N]
		similar_asins=[val[0].split('R_')[1] for val in most_sim]
		#now get the reviewtext and metadata based on the similar asin
		sim_images,sim_descriptions,sim_titles,sim_reviews=[],[],[],[]
		for sim_asin in similar_asins:
			#Note, change metadata_demo to metadata to get the full set, may be slower
			self.cur.execute("select a.metajson->'imUrl', a.metajson->'description', a.title, b.reviewCat from metadata_cell_phones_and_accessories a\
			join (SELECT asin,string_agg(reviewText,'. ') as reviewCat FROM reviews_cell_phones_and_accessories group by \
			asin) b on a.asin=b.asin where a.asin=%s limit 1;",(sim_asin,))
			result=self.cur.fetchall()[0]
			sim_images.append(result[0])
			sim_descriptions.append(result[1])
			sim_titles.append(result[2])
			sim_reviews.append(result[3])
		return similar_asins,sim_reviews,sim_images,sim_titles,sim_descriptions
		
	def q_filter(self,sentence):
		#filter the question text
		return [word.lower() for word in sum(self.process_line(sentence),[]) if word not in self.complete_bag]
		
	def q_filter_verb(self,sentence):
		return [word.lower() for word in sum(self.process_line(sentence),[]) if word not in self.complete_bag_verbs]
		
	def find_bigrams(self,key_words):
		ii=0
		while ii < len(key_words)-1:
			if key_words[ii]+'_'+key_words[ii+1] in self.QmodelB:
				key_words.insert(ii,key_words[ii]+' '+key_words[ii+1])
				key_words.pop(ii+1)
				key_words.pop(ii+1)
			ii+=1
		return key_words
		
	def return_key_words(self,question):
		question=question.lower()
		key_words_action= self.find_bigrams(self.q_filter_verb(question))
		key_words=self.find_bigrams(self.q_filter(question))
		[key_words_action.remove(word) for word in key_words]
		return key_words, key_words_action
		
	def find_relevent_sentence(self,text,key_words):
		text=text.lower()
		text=text.replace('/',' / ').replace('(',' ( ').replace(')',' ) ')
		good_sen=[]
		good_qual=[]
		good_qual_val=[]
		sentences = re.split(r"(?<![0-9])[.?!;](?![0-9])",text) #whatever delimiters you will need
		for sen in sentences: #this could also be done with regular expression
		#	if(set(key_words) & set(sen.split())): #find the intersection/union
		#		good_sen.append(sen)
		#		good_qual.append(str(len(set(key_words) & set(sen.split())))+'/'+str(len(set(key_words))))
		#		good_qual_val.append(len(set(key_words) & set(sen.split()))/len(set(key_words)))
		#########use regular expression to find the key words###############
			num_key_matches=len([re.findall(' '+word,sen) for word in key_words if len(re.findall(' '+word,sen))>0]) #number of key words matcheds
			if num_key_matches>0:
				good_sen.append(sen)
				good_qual.append(str(num_key_matches)+'/'+str(len(set(key_words))))
				good_qual_val.append(num_key_matches/len(set(key_words)))
			
		return good_sen,good_qual,good_qual_val
		
	def merge_review(self,sql_result):
		reviews=[]
		[reviews.append(review[0]) for review in sql_result]
		return' '.join(reviews)
		
	def check_key(self,word,model='question'):
		#return similar words based on the Question Bigram Model
		if model=='question':
			try:
				return self.QmodelB.most_similar(word,topn=5)
			except KeyError:
				return [['']]
		elif model=='review':
			try:
				return self.RmodelB.most_similar(word,topn=5)
			except KeyError:
				return [['']]
		else:
			return [['']]
	######### END SUPPORT FUNTIONS ####################################