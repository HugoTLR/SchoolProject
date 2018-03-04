import numpy as np
import time
from timeit import default_timer as timer

"""
	Return a pattern according to the original sign
	Split the signature in multiple part and get the mean value for each part.


"""
def create_pattern_from_sign(sign,patternsize=2):
	pattern = []

	sSize = len(sign)
	splitValue = int(np.ceil(sSize/patternsize))
	tmp = [-1,-1,-1]
	cpt = 0


	for i in range(splitValue):
		tmp[0] += sign[i][0]
		tmp[1] += sign[i][1]
		tmp[2] += sign[i][2]
		cpt += 1
	for i in range(0,3):
		tmp[i] = np.around(tmp[i] / cpt)
			
	pattern.append(tmp)
	tmp = [-1,-1,-1]
	cpt = 0

	for i in range(splitValue, sSize):
		tmp[0] += sign[i][0]
		tmp[1] += sign[i][1]
		tmp[2] += sign[i][2]
		cpt += 1
	for i in range(0,3):
		tmp[i] = np.around(tmp[i]/cpt)
			
	pattern.append(tmp)
	tmp = [-1,-1,-1]
	cpt = 0
	print(pattern)
	return pattern




"""
Compare a detected sign to a predifined pattern
Both parameters are list of lists
"""
def compare_with_patterns(pattern, sign, diffPercentage=12,minSimilarity=75):
	pSize = len(pattern)
	sSize = len(sign)
	#print("TAILLE SIGN: {}".format(sSize))
	splitValue = int(np.ceil(sSize/pSize))
	#print(splitValue)
	#print(format_sign(sign))
	resultList = []
	score = 0
	

	for i in range(splitValue):
		diffBlue = abs(pattern[0][0] - sign[i][0])
		diffGreen = abs(pattern[0][1] - sign[i][1])
		diffRed = abs(pattern[0][2] - sign[i][2])

		pctDiffBlue = round(diffBlue/255, 3)
		pctDiffGreen =  round(diffGreen/255, 3)
		pctDiffRed =  round(diffRed/255, 3)

		avgDiff = round((pctDiffRed + pctDiffGreen + pctDiffBlue) / 3 * 100, 3)


		if avgDiff <= diffPercentage: # Tweak this value to be more or less strict on diff
			score += 1

	resultList.append(score / splitValue *100)
	score = 0
	for i in range(splitValue, sSize):
		diffBlue = abs(pattern[0][0] - sign[i][0])
		diffGreen = abs(pattern[0][1] - sign[i][1])
		diffRed = abs(pattern[0][2] - sign[i][2])

		pctDiffBlue = round(diffBlue/255, 3)
		pctDiffGreen =  round(diffGreen/255, 3)
		pctDiffRed =  round(diffRed/255, 3)

		avgDiff = round((pctDiffRed + pctDiffGreen + pctDiffBlue) / 3 * 100, 3)

		#print("DIFF QUOTA:{} //// AVGDIFF: {}".format(diffPercentage, avgDiff))
		if avgDiff <= diffPercentage: # Tweak this value to be more or less strict on diff
			score += 1
	resultList.append(score / splitValue *100)
	if resultList[0] >= minSimilarity and resultList[1] >= minSimilarity :
		print(resultList)
		return True , resultList
	else:
		return False , resultList


"""
Create a signature from the image
Each line is filled with the mean color value of the line's pixels.
If save, return a np.array of 30 pixels wide to make a jpg image
"""
def GetSignature(frame,thresh,x,y,w,h,save=False):
	
	#start = timer()
	sign = []
	for j in range(y ,y+h-1):
		tmp = [-1,-1,-1]
		cpt = 1
		for i in range(x + int(round((w)/3)),x+w-1 - int(round((w)/3))):
			if thresh[j][i] != 0:

				tmp = tmp + frame[j][i]
				cpt = cpt + 1


		if tmp[0] and tmp[1] and tmp[2] != -1:
			tmp = tmp / cpt
			for i in range(0,3):
				tmp[i] = np.around(tmp[i])
			
			sign.append(tmp)


	if not sign:
		return
	if save:
		#SaveSignLog(sign)
		sign2 = np.zeros((len(sign),30,3), np.uint8)
		i = 0
		for d in sign:
			#print(d)
			sign2[i,:] = d
			i += 1
		sign = sign2

	#print("GetSignature: " + str(timer() - start))
	return sign


"""
Return a list of line pixel value from the original signature to a compressed one
"""
def createSign(signature, size):
	#start = timer()
	sign = []
	step = len(signature) / size
	cpt = 1
	mean = []
	tmp = [0,0,0]

	# Weird way to "create" data when needed for the program
	if len(signature) < size:
		i = 0
		j = 0
		while i < size:
			j += 1
			while i * step < j:
				sign.append(signature[j-1])
				i += 1
	else:
		i = 0.0
		for d in signature:

			mean.append(d)
			tmp = tmp + d
			i += 1
			if i >= cpt*step:
				tmp = tmp / len(mean)
				if len(sign) <= size:
					sign.append(np.around(tmp))
					cpt += 1
					mean.clear()
					tmp = [0,0,0]
				# else:
				# 	break
	
	#On colmate
	while len(sign) < size:
		sign.append(sign[-1]) 
	#print("createSign: " + str(timer() - start))
	return sign




"""
compare 2 signs and return its correspondance percentage
"""
def compare_signs(sign1, sign2, diffPercentage=12, minSimilarity=75):

	#start = timer()
	score = 0
	cpt = 0
	#print("Size data1: {} // Size data2: {}".format(len(data1),len(data2)))
	# if length <= 20:
	# 	print("\tABSOLUTE DIFF\t\t\tPCT DIFF \t\t\t\t AVG")

	for d in sign2:
		diffBlue = abs(d[0] - sign1[cpt][0])
		diffGreen = abs(d[1] - sign1[cpt][1])
		diffRed = abs(d[2] - sign1[cpt][2])

		pctDiffBlue = round(diffBlue/255, 3)
		pctDiffGreen =  round(diffGreen/255, 3)
		pctDiffRed =  round(diffRed/255, 3)

		avgDiff = round((pctDiffRed + pctDiffGreen + pctDiffBlue) / 3 * 100, 3)
		#if length <= 20:
			#print("B:{}\tG:{}\tR:{}\t\tpctB:{}\tpctG:{}\tpctR:{}\t\tAVG:{}".format(diffBlue,diffGreen,diffRed,pctDiffBlue ,pctDiffGreen,pctDiffRed, avgDiff))
		if avgDiff <= diffPercentage: # Tweak this value to be more or less strict on diff
			score += 1


		cpt +=1

	corres = score / len(sign2) *100


	#print("compare_signs: " + str(timer() - start))
	if corres >= minSimilarity :
		#print("Correspondance! {} %".format(corres))
		return True , corres
	else:
		return False , corres


"""
Format data
"""
def format_sign(signature):
	sign = str()
	first = True
	for item in signature:
		if first:
			first = False
		else:
			sign += "\n"

		sign += "{}:{}:{}".format(int(item[0]),int(item[1]),int(item[2]))
	return sign
