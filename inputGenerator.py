import random
from random import sample

def createSatClauses(numVariables, numClauses):
	clausesText = ""
	# get 1 .... numVariables as a list []
	variableList = list(range(1, numVariables + 1))
	# new line of text for each clause
	for currClauseNum in range(numClauses):
		#for each clause, we choose a random subset of variables to use ranging from 1 to the max possible (total number of variables)
		numVarsInCurrClause = 3
		# take a random subset of size ^ to use in our current clause
		listOfVarsInCurrClause = sample(variableList, numVarsInCurrClause)
		for var in listOfVarsInCurrClause:
			#  flip a coin if we should negate/not this variable in this clause or not
			negateOrNot = random.randint(0, 1)
			if(negateOrNot):
				negatedVar = var * -1
				clausesText += str(negatedVar) + " "
			else:
				clausesText += str(var) + " "
		clausesText += "\n"
	return clausesText

def createNonsatClauses(numVariables, numClauses):
	clausesText = ""
	# get 1 .... numVariables as a list []
	variableList = list(range(1, numVariables + 1))
	# new line of text for each clause
	for currClauseNum in range(numClauses):
		#for each clause, we choose a random subset of variables to use ranging from 1 to the max possible (total number of variables)
		numVarsInCurrClause = random.randint(1, numVariables)
		# take a random subset of size ^ to use in our current clause
		listOfVarsInCurrClause = sample(variableList, numVarsInCurrClause)
		for var in listOfVarsInCurrClause:
			#  flip a coin if we should negate/not this variable in this clause or not
			negateOrNot = random.randint(0, 1)
			if(negateOrNot):
				negatedVar = var * -1
				clausesText += str(negatedVar) + " "
			else:
				clausesText += str(var) + " "
		clausesText += "\n"

	return clausesText



def createTextFile(numVariables, numClauses, satisfiable):
	if(numVariables == 0 or numClauses == 0):
		return ""		
	finalText = str(numVariables) + " " + str(numClauses) + "\n"

	if(satisfiable):
		finalText += createSatClauses(numVariables, numClauses)
	else:
		finalText += createNonsatClauses(numVariables, numClauses)
	return finalText

if __name__ == '__main__':
	numVariables = input("Enter number of variables: ")
	numVariables = int(numVariables)
	numClauses = input("Enter number of clauses: ")
	numClauses = int(numClauses)
	satisfiable = input("Do you want this input satisfiable? (y/n)")
	if(satisfiable == "y"):
		satisfiable = True
	else:
		satisfiable = False

	outputText = createTextFile(numVariables, numClauses, satisfiable)
	outputFile = open("satInput.txt", "w")
	n = outputFile.write(outputText)
	outputFile.close()
	

