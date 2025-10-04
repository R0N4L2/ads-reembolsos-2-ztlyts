### Objective

Build a machine learning model to predict future match results.

### Brief

In this assignment, your task is to use historical information about soccer matches to build a machine-learning model for predicting future match results. Afterward, we will use that model to run multiple simulations of a Fifa World Cup tournament and produce statistics about which teams are the most likely to win it all.

### Tasks

Your task is to build a machine-learning model that can answer the following question:

-   Given a match between two teams, what is the _expected outcome_ at the end of the match?

With the trained model at our disposal, we want to run simulations for the 2018 World Cup (refer to ./qualified.csv).

-   Write a short program that simulates the entire 2018 tournament 1,000 times, calling into the model to get a prediction for each match.
-   Produce a short report on your findings, including statistics

### Data

The data in this repository ./matches.csv contains more than thirty-thousand international football matches played between 1950 and 2017. All matches are played between senior men's national teams - there are no club matches and no youth/women's games. You'll also find a ./teams.csv file containing information regarding the international associations and the FIFA confederations they are part of. You may find this information useful when looking at past opponents of a team. Last, we have a list of the teams that have qualified for the World Cup and their group stage draw in ./qualified.csv.

### Evaluation Criteria

-   Show us your work through your commit history.
-   Completeness: did you complete the features?
-   Correctness: does the functionality act in sensible, thought-out ways?
-   Maintainability: is it written in a clean, maintainable way?

### CodeSubmit

Please organize, design, test, and document your code as if it were going into production - then push your changes to the
master branch. After you have pushed your code, you may submit the assignment on the assignment page.

All the best and happy coding,

The ADS Team