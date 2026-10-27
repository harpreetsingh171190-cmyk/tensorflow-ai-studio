# LEGACY (2017): original LiveLessons environment (TensorFlow 1.0, Keras 2.0.8). Kept for reproducing the original lessons.
# For the modern API use Dockerfile.modern.
FROM jupyter/scipy-notebook

MAINTAINER Jon Krohn <jon@untapt.com>

USER $NB_USER

# install TensorFlow
RUN conda install --quiet --yes 'tensorflow=1.0*'

# install tflearn and keras: 
RUN pip install tflearn==0.3.2
RUN pip install keras==2.0.8

# install NLP packages:
RUN pip install nltk==3.2.4
RUN pip install gensim==2.3.0

# install Reinforcement Learning packages:
RUN pip install gym==0.9.4
