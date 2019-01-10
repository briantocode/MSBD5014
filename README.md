# MSBD5014

Readme


There are four models in the code folder. 
Run 'streamtrain_count.py' ,'streamtrain_speed.py','str_count_noweather.py' and 'str_speed_noweather.py' to get the four models, which is store in 'model' folder.

For each label to  predict,  one model is with weather features and the other doesnt  have weather feature. 

In model file, there are another two models.   They are not generate from the code in this folder. 

Run 'prediction_noweather.py' and 'prediction_withweather.py' to get the find preictions. all CSV files are stored in 'prediction' folder. 


environment:
python 3.6
packages:
sklearn
lightgbm
pandas
numpy
pickle
os
