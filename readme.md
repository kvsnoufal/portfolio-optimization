# Portfolio Optimization using Reinforcement Learning
Experimenting with RL for building optimal portfolio of 3 stocks and comparing it with portfolio theory based Markowitz' approach



Pls checkout the [medium article](https://medium.com/@noufalsamsudin/portfolio-optimization-using-reinforcement-learning-1b5eba5db072) for a quick overview.


To train RL model:
```
python train.py
```

To download data: 
1. https://www.mediafire.com/file/xivks3xf64b83ph/cleaned_preprocessed.csv/file
2. https://www.mediafire.com/file/g05yja1uiilhfuu/cleaned.csv/file

Take a look at pre_process.py if you want to get an idea on how this file was cleaned and compiled.


## Problem Statement

I will be formulating this as a portfolio optimization problem : 
Given histories of 3 different stocks, how would we allocate a fixed amount of money between these stocks every day so that maximize the likelihood of returns. 

The objective is to develop of policy (strategy) for building a portfolio. The portfolio is essentially an allocation of available resources across various stocks. The policy then needs to restructure the portfolio over time as new information becomes available.


![Pic of Model](https://github.com/kvsnoufal/portfolio-optimization/blob/main/img/po_model.png)


## RL agent training

![Pic of training](https://github.com/kvsnoufal/portfolio-optimization/blob/main/img/training.png)

## Results


![Pic of results](https://github.com/kvsnoufal/portfolio-optimization/blob/main/img/compare.png)











