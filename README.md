# Disaster Response Project

#### - Analyzing message data for disaster response -

#### Project description

The goal of this project is to build a solution that can classify text messages related to real-world disaster scenarios. 

In the case of a larger disaster thousands or even millions of messages are sent out in the form of social media messages, direct messages and news messages from different sources. Unfortunatly it is also at this time the disaster response organisations have the least capacity to filter, handle and take action on these messages. 

With the use of machine learning technology we can build a solution to classify text messages automatically. This could help to structure insights from these messages in a better way and in turn shift resources to where it is absolutely needed following a disaster.

The data used to make this possible are labeled direct messages, news messages and social media messages and comes from several different real world disasters and has been captured by Figure Eight.

#### The data
The data used to train my specific machine learning model are located in `$messages.csv` and `$catagories.csv`. These are combined and cleaned to get the labeled data that we use to train the machine learning model. 
 

#### Enhance, transform and load process
The python script `$process_data.py` located under the workspace/data folder can be run to load the data from the csv files, clean the data and to save the data to a database. 

The command to run the script looks like following: 

```sh
python process_data.py [messages file path] [categories file path] [output file path to database]
```
 

#### Training the machine learning model
The python script `$train_classifier.py` located under the workspace/models folder can be run to load the data from the database, tokenize the data, build the model, evaluated the model and then save the model in a pickle file. 

The command to run the script looks like the following:

```sh
python train_classifier.py [path to database] [output file path to pickle file]
```


#### Running the Web App
The web app is build with the python Flask framwork and can be run from the workspace/app directory. To start the app you need to go into the directory and write the following command.

```sh 
python3 run.py
```


