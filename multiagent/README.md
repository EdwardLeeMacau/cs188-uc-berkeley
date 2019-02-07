# Homework 2 strategy description

Student number: B05901119

**1 Basic evalFunction** 
Basically, I use the stateScore as the reference value.
This is reasonable, the score represent some siturations, such as: 
1. Eating the foods near the Pacman
2. Escaping from active pacman (Lose: -500, very negative expectation)
3. Eat capsule when the ghost is come near
4. Don't stop too long.

Another reason, appling the score is a way to represent that eating foods,scaredGhost are good action which these actions increase the score. (Not as Q1, no parameter <code>action</code> and I can't know whether the foods are eaten)

``evalScore = currentGameState.getScore()``

**2 Chasing the ghost**
Remember that eating capsules won't get any score, but given a potential to gain a large score(eat ghost or escape from lose). In here I would encourge Pacman to eat the capsules.

The weight should be higher... Because for each time eat a food get 10 point. I have tried 20 and 40, seems 40 gives a better expectation

```
# Eat the capsule
capsulePos = currentGameState.getCapsules()
if len(capsulePos) > 0:
    capsuleDistance = min([manhattanDistance(capsule, pacmanPos) for capsule in capsulePos])
    evalScore += 40.0 / capsuleDistance
```

After eaten capsules, it's able to eat the ghost. A concern of chasing far ghost is, wasting time an the scardTimer = 0 ... Therefore, set a easily evaluation to guess whether the Pacman can(or can't) eat the ghost (Define as canEat). I think <code>evalScore</code> can be a exponential decay function to the distance, and try 200, 100, 50, 25.

If the ghost is far from Pacman (>7), Pacman won't go because too many variance here.

```
# Chasing the scaredGhost
for ghostState in ghostStates:
    ghostPos = ghostState.getPosition()
    distance = manhattanDistance(ghostPos, pacmanPos)            

    canEat = int(ghostState.scaredTimer) / 2 > distance

    if canEat:
        if (distance <= 1):
            evalScore += 200
        elif (distance == 2):
            evalScore += 100
        elif (distance == 3):
            evalScore += 50
        elif (distance == 4):
            evalScore == 25
```

**3 Eat all the foods**
Sometimes Pacman have eaten all the foods around, and it can't see the further foods then stand here, encourage it to go here!
```
# Encourage to move and eat another food.
for foodPosition in foods:
    distance = manhattanDistance(foodPosition, pacmanPos)
    evalScore += 1.0 / distance
```

Something done not good is, Pacman don't know it's going into a dead end(be flanked). It needs to search and remember the entrance of the lane.
