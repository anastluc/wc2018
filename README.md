# wc2018

A simple data analysis of players of World Cup 2018 - more of a pandas exercise rather than a serious data analysis:)

## How to use

Install python requiremenets:
```
pip install -r requirements.txt
```
And start a notebook server:
```
jupyter notebook
```
Go to http://localhost:8888/notebooks/wc2016.ipynb 

## About

Get the players data from fifa world cup 2018 [official announcement pdf](https://img.fifa.com/image/upload/qcuxk3y7c1ezwo5yylnn.pdf)

Export the data (tables) using [tabula](https://tabula.technology/) and store it as csv.

And then do a simple analysis for simple stats like:
* age
  * average age of each national team
  * max (oldest), min (youngest)
  * age distribution
* height
  * average height of each team
  * max (tallest), min (shortest)
* bmi 
  * of each player (mass/height^2)
  * average of team 
  * max (fattest), min(lightest)
  * bmi distribution
  * bmi distribution by position (GK, DF, MF, FW)

* How many players play in domestic league per team

* Club representation
  What are the clubs that have the most players ?
  How do the numbers change as the tournament progresses (group-stage, round of 16, quarter finals ,..)

* Possible expansions (need merge with other datasets):

  * birthday paradox! (?)
    per match ? (what is the chance in 22 players to have the same birthday! - verify theoritical vs observation (group matches))
    (needs group matches info)

  * panini misses! 
    who are the players that panini missed or wrongly included?
    (needs panini album dataset - can be found [here (?)](https://www.cardboardconnection.com/2018-panini-world-cup-stickers-russia)

  * Team power ranking according to club ranking
    (needs club world ranking data)

  * Team value 
    [already here](https://www.transfermarkt.com/weltmeisterschaft-2018/startseite/pokalwettbewerb/WM18)
