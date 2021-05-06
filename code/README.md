# Flin Code Repository

This repository contains the code for [FLIN: A Flexible Natural Language Interface for Web Navigation](https://arxiv.org/abs/2010.12844) by Sahisnu Mazumder and Oriana Riva, which was accepted in the 2021 Conference of North American Chapter of the Association for Computational Linguistics (NAACL 2021).

If you use any of the materials in this repository, please cite the following paper.

```
bibtex
@inproceedings{flin21:naacl,
author = {Sahisnu Mazumder and Oriana Riva},
title = {{FLIN}: A Flexible Natural Language Interface for Web Navigation},
year = {2021},
booktitle = {Annual Conference of the North American Chapter of the Association for Computational Linguistics (NACCL 2021)},
eprint={2010.12844},
archivePrefix={arXiv},
address = {online},
month = jun
}
```

## About

The FLIN code can be found in the `code` folder. The code is organized in four directories:

1. `dataset_preparation` contains the code for reading website action parameter files, external resource files (e.g., list of dates in various formats) and data processing utility modules used by other modules in the repository. 

2. `navigation_module` contains the code to run the web navigation system which uses the trained AI agent. Given a natural language user utterance, the system returns a predicted navigation path.

3. `nsm_model` contains the code for building the AI agent. The AI agent is a deep learning model implemented using TensorFlow and BERT. 

4. `train_data_preprocessing` contains the code for data pre-processing and generating training and validation data. The data is saved in a vectorized dataset as `data_vec_dump.pickle` which is used for training the AI agent.

Additionally, in `code` there are four files that primarily serve as entry points to the above packages. 

1. `train.py` is the entry point to the `nsm_model` module which contains code for batch training of the AI agent using the generated `data_vec_dump` file. The entry point of train_5.py is main.py which invokes training functions.

2. `evaluation.py` is the entry point to `navigation_module` which contains code for running the AI agent on a test trace in various evaluation modes. This code is invoked by `main.py` which collects the evaluation modes (the `eval_mode` argument) as specified by the user.

3. `globals.py` contains global variables.

4. `main.py` is the main entry point of the full code repository. If a trained model exists in the `qa_model` folder, it simply bypasses the model training and invokes the evaluation module. Otherwise, it trains the model using `data_vec_dump.pickle` and starts the evaluation.

	`main.py` takes various command line arguments. The important ones include:
	- `test-trace-id` specifying which test trace the evaluation has to be performed on.
	- `eval-mode` specifying the executon mode of the agent. `eval-mode`  can take two values:
		- `of`: the evaluation runs in offline mode on the test queries and evaluation results are obtained by comparing the predicted paths against the gold paths
		- `i`: the agent runs in interactive mode where a user can manually issue test queries and see the prediction results
		
The `all_results` folder contains all evaluation results for FLIN and its variants FLIN-sem (based on semantic similarity only) and FLIN-lex (based on lexical similarity only) reported in our paper. 


# Disclaimer
In our implementation, we built a graph DB for each website and hosted them in [Azure Cosmos DB] (https://docs.microsoft.com/en-us/azure/cosmos-db/graph-introduction)
for analysis and query purposes (using the Gremlin API). The code repository (in parcular evaluation.py) was linked to the graph DB for real-time quering of website 
schema and utilized it for web navigation. In the realese, we are currently unable to provide access to our hosted graph DB and its code interface to run the code with remote 
graph server access facility. Thus, in order to run the code in `eval-mode`, we recommend users to write a website schema reading module to be interfaced with the code 
(see Line # 20 - 24 in evaluation.py) to make it runnable. 

The website schema reading module should read the [trace_id]_action.csv and [trace_id]_para.csv files in [trace_id]_datafiles/ folders under webnav_dataset/ directrory to fetch 
information about the correspoding website schema ( i.e, states, actions, parameters and their domains) and  index and store them in local data structures like dictionaries for 
in-memory usage (e.g., node_DB for states and actions, para_DB for action parameters and para_dom for parameter domains) to be used by other modules in the navigation_module/ package.
We will update the code repository to incorporate the website schema reading module for running the code locally sooner.

## Set up

- Install Python 3.6. The code is developed using Python 3.6.8. Python 3.7 may cause compatibility issues.
- Install the required Python packages
    ```
	bash
	cd code/
    pip install -r requirements.txt
    ```
- Install additional resources for NLTK. In a Python console:
    ```
	bash
    $ python3.5

    >> import nltk
    >> nltk.download('stopwords')
    >> nltk.download('averaged_perceptron_tagger')
    >> nltk.download('wordnet')
    ```
- Once done, run the code as follows to verify the setup
    ```
	bash
    cd code/
    python main.py –eval-mode=of
    ```

    If the code runs successfully, and there is no import error, then everything is set up properly 
    and you are ready to go. 

