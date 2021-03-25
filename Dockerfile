FROM python:3.8-slim

WORKDIR /AppleTree 

COPY requirements.txt /

COPY  DecisionTreePredictor.py . /AppleTree/

RUN pip install -r /requirements.txt

CMD [ "python", "./DecisionTreePredictor.py" ]