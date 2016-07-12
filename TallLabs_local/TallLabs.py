#import os
#base_dir=os.path.expanduser('~')
from __future__ import division
from flask import Flask, render_template, request, request, jsonify
import numpy as np
from TallLabs_class import *

#initialize emoji class
Tall=TallLabs_lib()

#Setup Authentication #########################################
from functools import wraps
from flask import request, Response

def check_auth(username, password):
	"""This function is called to check if a username /
	password combination is valid.
	"""
	return username == 'insight' and password == 'demo'

def authenticate():
	"""Sends a 401 response that enables basic auth"""
	return Response(
	'Could not verify your access level for that URL.\n'
	'You have to login with proper credentials', 401,
	{'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
	@wraps(f)
	def decorated(*args, **kwargs):
		auth = request.authorization
		if not auth or not check_auth(auth.username, auth.password):
			return authenticate()
		return f(*args, **kwargs)
	return decorated
############# End Authentication ############################

application = Flask(__name__)

####### Tall Labs Pages ######################

##### Make a homepage by setting one of the pages to index.html
#@ application.route("/")
#def index():
#	return render_template("index.html")

@application.route("/force")
def force():
	return render_template("force.html")
@application.route("/tree")
def tree():
	return render_template("tree.html")
@application.route("/tree_doc")
def tree_doc():
	return render_template("tree_doc.html")
@application.route("/train")
def train():
	return render_template("train.html")
@application.route("/slides")
def slides():
	return render_template("slides.html")
@application.route("/demo")
#@requires_auth
def demo():
	return render_template("search.html")
	
#Functions for D3JS visualization
@application.route('/_get_vis')
def _get_vis():
	word = request.args.get('word')
	model = request.args.get('model')
	result=Tall.visual(word,model)
	return jsonify(result=result)
	
@application.route('/_get_tree')
def _get_tree():
	word = request.args.get('word')
	model = request.args.get('model')
	result=Tall.tree(word,model)
	return jsonify(result=result)
	
@application.route('/_get_tree_doc')
def _get_tree_doc():
	asin = request.args.get('asin')
	keys = request.args.get('keys')
	result=Tall.tree_Doc(asin,keys)
	return jsonify(result=result)

#Functions for mechanical turck
@application.route('/_train')
def _train():
	res=request.args.get('a')
	name=request.args.get('name')
	result=Tall.train(res,name)
	return jsonify(result=result)
	
@application.route('/_train_plot')
def _train_plot():
	result=Tall.train_plot()
	return jsonify(result=result)
	
@application.route('/_train_table')
def _train_table():
	result=Tall.leader_board()
	print(result)
	return jsonify(result)

#Functions for webapp demo
@application.route('/_update_item')
def _update_item():
	asin = request.args.get('asin')
	image,title,description,ques,revs=Tall.getMeta(asin)
	return jsonify(image=image,title=title,desc=description,ques=ques,revs=revs)
	
@application.route('/_process_question')
def _process_question():
	question = request.args.get('question')
	asin = request.args.get('asin')
	answers,about_text,similar_answer,title2,image_url2=Tall.processQuestion(asin,question)
	return jsonify(result=answers,about=about_text,similar=similar_answer,title2=title2,image2=image_url2)
	
###### End Tall Labs ########################
if __name__ == "__main__":
	application.run(host='0.0.0.0')
