# Disaster Response Pipeline Project
### Running instructions for linux environment
#### Prerequisite to have been installed.
**01.** Python3.6(Python3)  
**02.** Virtualenv  
**03.** Requirements that are in **requirements.txt**  

##### Step 01: Clone the project to your workplace
If you are using a ssh key for GitHub, please use flowing git command to clone the project.
```
git clone git@github.com:snandasena/disaster-response-pipeline.git
``` 
Otherwise use HTTPS git method to clone the git repositories.  
```
git clone https://github.com/snandasena/disaster-response-pipeline.git
```  
##### Step 02: Create a Python3.6(Python3) virtual environment inside the project root directory
Here I assumed you are in the project root directory and here **"."** is represent the current directory.  
```
virtualenv -p python3.6 .
``` 
Then activate the python virtual environment
```
source bin/active
```
#### Run application with a single command in linux
```bash 
chmod 755 run.sh
./run.sh
```
#### Description for ``run.sh``
```bash
#!/bin/bash
# to install required python dependencies
python -m pip install -r requirements.txt

# to install nltk packages
python -m nltk.downloader wordnet
python -m nltk.downloader stopwords
python -m nltk.downloader punkt

# to run ETL pipeline
rm -rf ./data/DisasterResponse.db >/dev/null
#
python process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv ./data/DisasterResponse.db

# to run ML pipeline

rm -rf ./models/classifier.pkl >/dev/null
python train_classifier.py ./data/DisasterResponse.db ./models/classifier.pkl

# run the application
# shellcheck disable=SC2164
python run.py ./data/DisasterResponse.db ./models/classifier.pkl
```
#### Note: To enable hyper parameter tuning please change the following code line 
**From:** 
```python
model = build_model()
```
**To:**
```python        
model = build_model(enable_param_tuning=True)
```
####  in ``train_classifier.py``
 
After ETL and ML pipeline completion, the app will be started automatically.  
Then go to web browser and hit the this url ``http://localhost:3001/``.

### Acknowledgments  
**Licensing, Authors, Acknowledgements**

Big thank you to [Udacity](https://www.figure-eight.com/) for providing the template code for this project. Also want to thank [Figure Eight](https://www.figure-eight.com/) for providing the data.  
