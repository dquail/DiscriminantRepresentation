import numpy as np
import random
import matplotlib.pyplot as plt

'''
Simple MDP:
2 states (0 and 1). 0 is represented by n bias bits and a 1 at the end. 1 is represented by n bias bits and a 0 at the end.
The 'M'ove action transitions between the two and returns a reward of 0. 
The 'T'ouch action will return a reward of 0 if in state 0 and and 0 if in state 1.    
'''


class Foreground:

    def __init__(self, numberOfBiasBits=5, stepsBeforeSwitching=1, totalSteps=10000, alpha = 0.1, pctTimeInState0 = 0.33, pctTimeInState1 = 0.33):
        if pctTimeInState1 + pctTimeInState0 > 1.0:
            raise Exception("Error: pctTimeInState1 + pcTimeInState0 must be less than 1.0")

        self.pctTimeInState0 = pctTimeInState0
        self.pctTimeInState1 = pctTimeInState1
        self.alpha = alpha / (numberOfBiasBits + 1)
        self.numberOfBiasBits = numberOfBiasBits
        self.stepsBeforeSwitching = stepsBeforeSwitching
        self.stepsSinceSwitching = 0
        self.totalSteps = totalSteps
        self.currentStateIndex = 0
        self.states = []
        self.state0Predictions = []
        self.state1Predictions = []

        # Set up first state
        arr0 = np.ones(numberOfBiasBits + 1)
        self.states.append(arr0)

        # Set up second state
        arr1 = np.ones(numberOfBiasBits + 1)
        arr1[numberOfBiasBits] = 0.0
        self.states.append(arr1)

        #Set up weight vector
        self.weights = np.zeros(self.numberOfBiasBits + 1)

    def getAction(self):
        if self.stepsSinceSwitching < self.stepsBeforeSwitching:
            self.stepsSinceSwitching += 1
            return 'T'
        else:
            self.stepsSinceSwitching = 0
            return 'M'

    def takeStep(self, action):
        if action == 'T':
            if self.currentStateIndex == 0:
                reward = 1
                self.currentStateIndex = 0
            elif self.currentStateIndex == 1:
                reward = 0
                self.currentStateIndex = 1
        if action == 'M':
            reward = 0
            if self.currentStateIndex == 0:
                self.currentStateIndex = 1
            elif self.currentStateIndex == 1:
                self.currentStateIndex = 0
        state = self.states[self.currentStateIndex]

        return reward, state

    def prediction(self, state):
        return np.inner(self.weights, state)

    def learn(self, action, reward, state, nextState):
        #learn from this
        if action == 'T':
            error = reward - self.prediction(state)
            self.weights = self.weights + self.alpha * error * state
        #Otherwise there's nothing to learn from since action is to move


    def testHowLongToNoError(self):
        #Determines how long until there is virtually no error in both of the predictions

    '''
    More iid RL agent. Able to warp between states
    '''
    def startWithWarp(self):
        print("Starting")
        for step in range(self.totalSteps):
            if step % 100:
                print("Processing " + str(step) + " .....")
                #get action
                i = random.uniform(0,1)
                if i < self.pctTimeInState0:
                    #Go to state 0 and perform 'T'
                    self.currentStateIndex = 0
                    previousState = self.states[0]
                    reward, currentState = self.takeStep('T')
                    self.learn('T', reward, previousState, currentState)
                elif i < self.pctTimeInState0 + self.pctTimeInState1:
                    self.currentStateIndex = 1
                    previousState = self.states[1]
                    reward, currentState = self.takeStep('T')
                    self.learn('T', reward, previousState, currentState)
                # Update the prediction arrays
                prediction = self.prediction(self.states[0])
                self.state0Predictions.append(prediction)

                prediction = self.prediction(self.states[1])
                self.state1Predictions.append(prediction)

        # Display the prediction array.
        self.plotPredictions()

    '''
    Traditional RL agent. Going from state to state via actions and learning
    '''
    def start(self):
        print("Starting ... ")
        for step in range(self.totalSteps):
            if step % 100:
                print("Processing " + str(step) + " ..... ")
            action = self.getAction()
            previousState = self.states[self.currentStateIndex]
            reward, currentState = self.takeStep(action)

            self.learn(action, reward, previousState, currentState)

            #Update the prediction arrays
            prediction = self.prediction(self.states[0])
            self.state0Predictions.append(prediction)

            prediction = self.prediction(self.states[1])
            self.state1Predictions.append(prediction)

        #Display the prediction array.
        self.plotPredictions()

    def plotPredictions(self):
        print("Plotting")

        fig = plt.figure(1)
        fig.suptitle('Bandit', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(211)
        titleLabel = "BIAS exoeriment"
        ax.set_title(titleLabel)
        ax.set_xlabel('Step')
        ax.set_ylabel('In corner prediction')

        ax.plot(self.state0Predictions, label = "Corner")
        ax.plot(self.state1Predictions, label = "Wall")
        plt.show()


#foreground = Foreground(numberOfBiasBits = 100, totalSteps= 400000, pctTimeInState0 = 0.1, pctTimeInState1 = 0.01)
#foreground.start()
#foreground.startWithWarp()