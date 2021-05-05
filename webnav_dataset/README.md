# WebNav dataset

This repository contains the dataset for "FLIN: A Flexible Natural Language Interface for Web Navigation" by Sahisnu Mazumder and Oriana Riva, which was accepted in the 2021 Conference of North American Chapter of the Association for Computational Linguistics (NAACL 2021).

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


## Data description

The dataset contains training, validation and test data for action graphs extracted from 9 popular websites. The `website_index.txt` file contains the mapping between the website id (websiteID) and the website URL.

### Parameter and action data formats

Each `[websiteID]_datafiles` folder contains `[websiteID]_para.csv` and `[websiteID]_action.csv` files for each of the 9 websites. `[websiteID]_para.csv` file provides the parameters for each action supported by the website which can be found in the corresponding `[websiteID]_action.csv` file. 

Each row of `[websiteID]_para.csv` has the following format:
```
[source_state]->[destination_state]#[action_name],[parameter_name],[domain_value],[parameter_type]
```
Each row of `[websiteID]_action.csv` has the following format:
```
SD: [source_state]###[destination_state],PL:[path_length],PATH: [source_state]->[destination_state]#[action_name][[parameter1]:[parameter2]:....:[parameterK]]
```

#### Example

The top-row in `925_para.csv` is 

```
/s/->restaurant#restaurant,restaurant,520 bar grill,CLOSED"
```
where "/s/" is the source_state (a search results webpage) and "restaurant" is the destination_state (a restaurant info page), and action name and parameter name are both "restaurant". The parameter is a closed-domain parameter and has a domain value "520 bar grill" which is a name of a restaurant in the search page ("/s/").

Action information can be found in `925_action.csv` [Line 13]: 
```
SD: /s/###restaurant, PL: 1, PATH: /s/->restaurant#restaurant[]
```

where [source_state] is "/s/" and [destination_state] is "restaurant". The execution path specified in "PATH:..." causes a transition from a search ("/s/") page to a restaurant ("restaurant") page and the length of the path for the transition is 1, i.e a 1-step transition denoted by "PL:1". The execution path "/s/->restaurant#restaurant[]" corresponds to selecting a restaurant in a list of search results and transitioning to the restaurant profile page.

The action_name "restaurant" does not have any associated parameters. Thus the parameter list (":" separated) denoted by [[parameter1]:[parameter2]:....:[parameterK]]
is empty. For action with empty list of parameters, we introduce a dummy parameter which is named same as the action_name in `[websiteID]_para.csv`, for uniformity in format. In the above example, we consider the "restaurant[]" action has a parameter_name called "restaurant" (which represents the name of the selected restaurant) introduced in `925_para.csv` as explained above.

NOTE: The domain values of all date parameters in `[websiteID]_para.csv` are loaded from `Dates.csv` which contains a long list of dates. We use the 2nd column of `Dates.csv` as our data parameter domain values.

### Training/Validation/Test files

There are 9 `[websiteID]_datafiles/` folders, one for each website. Each `[websiteID]_datafiles/` folder has three files of the format: `[websiteID]_train_q.csv` (for training), `[websiteID]_valid_q.csv` (for validation) and `[websiteID]_test_q.csv` (for testing).

Each line of these train/valid/test csv files contains a natural language (NL) user command (or query) and one or more (gold) navigation instruction for the command, in the following format:

```
[Query], [path1] || [path2] ....
```

where [Query] is the user NL query and [path1], [path2] ... is list of gold navigation paths for that query. Query and paths are separated by comma (‘,’).

[path1] has the following format:

```
[source state]->[destination state]#[action_name]{ [parameter name] = [parameter value] | [parameter type] | [parameter mention]; . . .}
```

The navigation instruction is composed of website actions and parameters with their value assignments which can be interpreted using the information in `[websiteID]_action.csv` and `[websiteID]_para.csv`. Specifically:

- “[parameter value]” is the gold domain value to be mapped, 
- “[parameter type]” can be 1 (closed-domain) or 0 (open-domain) as specified in the CLOSED/OPEN column values in `[websiteID]_para.csv`, and 
- “[parameter mention]” is the mention phrase of parameter specified by “[parameter name]” in the query.

#### Example

We consider the the following data line: 
```
restaurant for a party of 9 people,/->/s/#let s go{'people'= '9 people'|1|'a party of 9 people')}||/s/->/s/#find table{'people'= '9 people'|1|'a party of 9 people')}
```

- Query = “restaurant for a party of 9 people”
- [path1] = /->/s/#let s go{'people'= '9 people'|1|'a party of 9 people')}
- [path2] = /s/->/s/#find table{'people'= '9 people'|1|'a party of 9 people')}

Path1 and Path2 are two gold navigation paths for the query. Path1 indicates that, given the homepage (‘/’) of Opentable.com, the system should click on action “let s go” with parameter 'people'= '9 people' and the execution of the action will lead to the search result page (‘/s/’). The notation '9 people'|1|'a party of 9 people' denotes 'people'= '9 people' (domain value) where ‘people’ is a closed-domain parameter and 'a party of 9 people' is the mention for parameter ’people’ in query “restaurant for a party of 9 people”.

## Disclaimer

Query examples in this dataset that appear to be attributed to a user or related user contact information are for illustration only and are fictitious. No real association is intended or inferred.