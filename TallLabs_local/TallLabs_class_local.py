#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division,unicode_literals
import os
base_dir=os.path.expanduser('~')+'/TallLabs' #get home dir, point to emojify folder
import datetime
import numpy as np
import re, collections, random, datetime
import psycopg2
import json
from gensim import corpora, models, similarities

# guarantee unicode string... #no need in python3 (will cause an error)
_u = lambda t: t.decode('UTF-8', 'replace') if isinstance(t, str) else t

class TallLabs_lib:
	"""Tall Labs Class"""
	def __init__(self):
		#setup
		self.conn = psycopg2.connect("host=localhost port=5432 dbname=qa") #called amazon on aws
		self.cur = self.conn.cursor()
		self.stoplist = set('a an for of the and to in rt'.split())
		self.QmodelB=models.Word2Vec.load('models/QmodelB')
		self.RmodelB=models.Word2Vec.load('models/QmodelB')#modify for aws to look for bigram model
		self.bag_of_words_yn='is,will,wil,may,might,does,dose,doe,dos,do,can,could,must,should,are,would,do,did'.split(',')
		self.bag_of_words_oe="what,what's,where".split(',')
		self.bag_of_words='is,will,wil,may,might,does,do,can,could,must,should,are,would,did,need,take,out,how,would,am,at,\
anyone,has,have,off,that,which,who,please,thank,you,that,fit,these,they,many,work,with,time,turn,fit,fitt,\
from,hard,use,your,not,into,non,hold,say,from,one,two,like,than,same,thanks,find,make,hot,be,as,well,there,\
son,daughter,amazon,when,after,change,both,ask,know,help,me,recently,purchased,item,any,newest,or'.split(',')
		self.bag_of_words_verbs='is,will,wil,may,might,does,do,can,could,must,should,are,would,did,take,out,would,\
anyone,off,that,which,who,please,thank,you,that,these,they,many,time,turn,newest,there,am,at,\
from,hard,use,your,not,into,non,hold,say,from,one,two,like,than,same,thanks,\
son,daughter,amazon,when,after,change,both,ask,know,help,me,recently,purchased,item,any'.split(',')
		self.complete_bag=set(sum([[item[0] for item in self.QmodelB.most_similar(word)] for word in self.bag_of_words],[]))|self.stoplist|set(self.bag_of_words)
		self.complete_bag_verbs=set(sum([[item[0] for item in self.QmodelB.most_similar(word)] for word in self.bag_of_words_verbs],[]))|self.stoplist|set(self.bag_of_words_verbs)
		
		
	def clean_result(self,model_result):
		return [item[0] for item in model_result],[item[1] for item in model_result]
		#topn=15
	def visual(self,word):
		#print(type(self.QmodelB))
		word=word.lower()
		results,counts=self.clean_result(self.QmodelB.most_similar(word,topn=5))
		source_target=[]
		for result_word in results:
			#append source target, and search next layer
			source_target.append((word,result_word,1))
			results2,counts2=self.clean_result(self.QmodelB.most_similar(result_word,topn=5))
			for result_word2 in results2:
				source_target.append((result_word,result_word2,2))
				results3,counts3=self.clean_result(self.QmodelB.most_similar(result_word2,topn=5))
				for result_word3 in results3:
					source_target.append((result_word2,result_word3,3))
		return [{"source":src,"target":tar,"group":grp} for src,tar,grp in source_target]
		
	def tree(self, word):
		word=word.lower()
		results,counts=self.clean_result(self.QmodelB.most_similar(word))
		target=[]
		child_list=[]
		for result_word in results:
			target_child=[]
			target.append(result_word)
			results2,counts2=self.clean_result(self.QmodelB.most_similar(result_word))
			for result_word2 in results2:
				target_child.append(result_word2)
			child_list.append(target_child)
		return {"name":word,"children":[{"name":tar,"children":[{"name":child,"size":3} for child in child_l] }\
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
		#step 1, split
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
		self.cur.execute("select metajson->'imUrl', metajson->'description', title from metadata where asin=%s and id >1000000 limit 1;",(asin,))
		self.result=cur.fetchall()[0]
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
		
		return image,description,title,question,formated_reviews
		
	def processQuestion(self,asin,question):
		key_words, key_words_action = self.return_key_words(question)
		#print(key_words,key_words_action)
		similar_keys=sum([[' '.join(item[0].split('_')) for item in self.check_key(word,'review') if item!=[''] and item[1]>0.7]\
		for word in key_words],[])
		### pull review data
		self.cur.execute("select reviewtext from reviews_cell_phones_and_accessories where asin=%s;",(asin,))
		result=self.cur.fetchall()

		good_sen,good_qual,good_qual_val=self.find_relevent_sentence(self.merge_review(result),key_words)
		#print(good_qual_val)
		sorted_index=sorted(range(len(good_qual_val)),key=lambda x:good_qual_val[x])[::-1]
		
		formatted_answer='\n\n'.join([good_qual[index]+':'+good_sen[index] for index in sorted_index][0:5])
		
		about_text='Question Type: '+'Yes/No' + '\n'+\
		'Key Words = '+ ', '.join(key_words) + '\n'+\
		'Action Words = '+ ', '.join(key_words_action) + '\n'+\
		'Similar Keys = '+ ', '.join(similar_keys)
		
		return formatted_answer, about_text
		
	###### Support functions ########################################################################
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
		for sen in sentences:
			if(set(key_words) & set(sen.split())): #find the intersection/union
				good_sen.append(sen)
				good_qual.append(str(len(set(key_words) & set(sen.split())))+'/'+str(len(set(key_words))))
				good_qual_val.append(len(set(key_words) & set(sen.split()))/len(set(key_words)))
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