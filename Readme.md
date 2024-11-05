# Building a character level language model

- Create a `venv` and install all required packages from `requirements.txt`
run  
```
python -m venv venv

.\venv\Scripts\activate 
or
venv/bin/activate

pip install -r ./requirements.txt
``` 
- To run bigram model to generate names go to bigram folder and run

```
python run.py 5
```

we can change the num of names by changing the '5' in cli with number of choice

- To train model we can run

```
python train.py
```

This will train the model with data from `data\IndianNames.txt` and after model is trained it will save the model as onnx in models folder.

I chose onnx because i wanted to create frontend for this as well. And in frontend I used onnx and tfjs to import and run the model and give output. I can make backend api seperately and make a frontend api request but for that i have to write backend and deploy it so to escape from that effort and explore tfjs and onnx i made it this way.Frontend is deployed in vercel. For anyone who is interested this is the site link [Indian Name Generator](https://india-name-generator.vercel.app/). And this is the repo link [Repo](https://github.com/uvamsi76/namegenerator) 

I have hardcoded everything in train.py like the data location. place to store model.learning rate etc etc.I have written the code very badly I will package everything properly in the future but for now this is it.

I am doing this along side a video of andrej karpathy's ***makemore***. This project will be mostly experimenting with things, trying to extend a bit more from the videos and add something based on what i have learned and what i know.  
