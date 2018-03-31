# import the necessary packages
import argparse
import numpy as np
from imutils.object_detection import non_max_suppression
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import datetime
import imutils
import time
from timeit import default_timer as timer
import datetime
import operator
import cv2 as cv
import sqlite3
import random
import io
import Signature as sign
import json


"""
Create json file with all sign detected during session
"""
def createJsonLog(signlist):
	data = []
	cpt = 1
	for s in signList:
		data.append({'id': cpt, 'sign' : []})
		signat = sign.createSign(s,100)
		for val in signat:

			data[-1]['sign'].append(val.tolist())
		cpt = cpt +1
	filename = "session_"+time.strftime("%d-%m-%Y")+'-'+time.strftime("%H-%M-%S")+".txt"
	with open('Log/'+filename, 'w') as outfile:
		json.dump(data, outfile)



"""
Create SQL TABLE
"""
def create_table():
	curs.execute('CREATE TABLE IF NOT EXISTS people(signid INTEGER PRIMARY KEY,datestamp TEXT, smallsign TEXT, bigsign TEXT)')

"""
Read db example
"""
def read_from_db():
    curs.execute("SELECT datestamp,smallsign,bigsign FROM people")
    data = curs.fetchall()
    for row in data:
        print("{}\n\n{}\n\n\n\n".format(row[1],row[2]))
        # cv.imwrite("./Images/"+row[0]+"-small.jpg")
        # cv.imwrite("./Images/"+row[0]+"-big.jpg")

"""
Check if db empty
"""
def IsDbEmpty():
	curs.execute("SELECT * FROM people")
	if curs.fetchone() is None:
		return True
	else:
		return False

"""
Return similar signs from db
"""
def get_similar_signs(smallsign,bigsign,diffPercentage=12,minSimilarity=75):
	start = timer()
	curs.execute("SELECT datestamp,smallsign,bigsign FROM people")
	data = curs.fetchall()

	nbSmallCorres = 0
	nbBigCorres = 0

	similarSmallSign = []
	similarBigSign = []
	for row in data:
		if sign.compare_signs(smallsign,convertItem(row[1]),diffPercentage)[1] >= minSimilarity:
			nbSmallCorres += 1
			similarSmallSign.append((row[0],row[1]))
		if sign.compare_signs(bigsign,convertItem(row[2]),diffPercentage)[1] >= minSimilarity:
			nbBigCorres += 1
			similarBigSign.append((row[0],row[2]))

	print("{} similar small sign in DB // {} similare big sign in DB".format(nbSmallCorres,nbBigCorres))


	#print("getSimilarSign: " + str(timer() - start))
	return similarSmallSign,similarBigSign



"""
Insert signs into DB
"""
def dynamic_data_entry(smallSign, bigSign):
	unix = int(time.time())
	datest = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S'))

	curs.execute("INSERT INTO people (datestamp, smallsign, bigsign) VALUES (?,?,?)",
		(datest, smallSign,bigSign))
	conn.commit()


"""
Get a sign from db and create a simple pattern
itemID = id in db
type: 0 small, 1 big
"""
def get_sign_from_db(itemID=3, stype=0):
	if stype == 0:
		curs.execute("""SELECT signid,datestamp,smallsign FROM people WHERE signid=?""",(itemID,))
	else:
		curs.execute("""SELECT signid,datestamp,bigsign FROM people WHERE signid=?""",(itemID,))

	data = curs.fetchone()
	
	return convertItem(data[2])


"""
Convert string data to list of list [[b,g,r],[b,g,r]]
"""
def convertItem(item):
	res = []
	for i in item.split('\n'):
		b, g, r = i.split(':')
		res.append([int(b),int(g),int(r)])
	return res

"""
Create an image of the sign to make comparison easier
"""
def createSignPreview(sign,name="Sign Preview",save=False):
	i = 0
	if len(sign) == 2:
		blank_image = np.zeros((20,30,3), np.uint8)
		for i in range(20):
			#print(d)
			if i <= 9:
				blank_image[i,:] = sign[0]
			else:
				blank_image[i,:] = sign[1]

			i += 1

	else:
		blank_image = np.zeros((len(sign),30,3), np.uint8)
		for d in sign:
			#print(d)
			blank_image[i,:] = d
			i += 1

	cv.imshow( name, blank_image)
	if save:
		cv.imwrite("./Images/"+name+ str(dvatetime.date.today()) + ".jpg" ,blank_image)
	#return blank_image


"""
BD STUFF
connect + grab a signature wich will be our target
"""
conn = sqlite3.connect('humantracking.db')
curs = conn.cursor()
#read_from_db()





# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
ap.add_argument("-f", "--feed-db", type=bool, default=False, help="if true, will save detected sign on DB")
args = vars(ap.parse_args())

feed = args["feed_db"]
b = False
hasTarget = False

signList = []

if IsDbEmpty():
    feed = True
else:
	targetSign = get_sign_from_db(3,1)
	targetPattern = sign.create_pattern_from_sign(targetSign)
	hasTarget = True


	"""
	User Interface
	Show the sign and its pattern
	"""
	createSignPreview(targetSign)
	createSignPreview(targetPattern,"Pattern")

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv.VideoCapture(args["video"])


# initialize the first frame in the video stream
firstFrame = None

detected = False

fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

# initialize the HOG descriptor/person detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

endtime = 0.0

# loop over the frames of the video
while True:

	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break


	
	diff = endtime - timer() 
	if diff >= 200:
		detected = False
		#print(diff)
		endtime = 0.0

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	frame = cv.flip(frame,0)

	
	fgmask = fgbg.apply(frame)

	# if the first frame is None, initialize it
	if firstFrame is None:
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		gray = cv.GaussianBlur(gray, (43, 43), 0)
		firstFrame = gray
		continue

	orig = frame.copy()
 
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
		padding=(16, 16), scale=1.05)
 
	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 	
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	if len(pick) > 0:
		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			if yB >= 100 and xB <= 0.75*yB:
				cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
				#print("xA:{},yA:{},xB:{},yB:{}".format(xA,yA,xB,yB))
				signature = sign.GetSignature(frame,fgmask,xA,yA,xB,yB)

				signList.append(signature)
				"""
				TODO: Find a way to detect if sign is worth to check
				For exemple: something that cross the field of the camera too close
				"""
				if signature is not None:
					bigsign = sign.createSign(signature,100)
					if hasTarget is True:
						b, val = sign.compare_signs(targetSign,bigsign, 5,85) #diffPercentage, minSimilarity

					if not detected and b:
						detected = True
						print("TARGET FOUND !!!")
						roi = frame[y:y+h, x:x+w]
						cv.imwrite("./Images/FOUND-"+str(endtime)+"_"+str(val)+".jpg", roi)
					elif not detected and feed is True:
						detected = True
						smallsign = sign.createSign(signature,100)
						small = sign.format_sign(smallsign)
						big = sign.format_sign(bigsign)
						smallList,bigList = get_similar_signs(smallsign,bigsign)
						if len(smallList)+len(bigList) <= 5:
							dynamic_data_entry(small,big)


	# fgmask = fgbg.apply(frame)
	# (_, cnts, _) = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	# (x, y, w, h) = (0,0,0,0)
	# # loop over the contours
	# for c in cnts:
	# 	# if the contour is too small, ignore it
	# 	if cv.contourArea(c) < args["min_area"]:
	# 		continue
 
	# 	# compute the bounding box for the contour, draw it on the frame,
	# 	(x, y, w, h) = cv.boundingRect(c)

	# 	#Make sure the detection is tall enough to fill our 'bigSign' wich is 100
	# 	if h >= 100 and w <= 0.75*h:
	# 		cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# 		signature = sign.GetSignature(frame,fgmask,x,y,w,h)


	# 		"""
	# 		TODO: Find a way to detect if sign is worth to check
	# 		For exemple: something that cross the field of the camera too close
	# 		"""
	# 		bigsign = sign.createSign(signature,100)
	# 		if hasTarget is True:
	# 			b, val = sign.compare_signs(targetSign,bigsign, 5,85) #diffPercentage, minSimilarity

	# 		if not detected and b:
	# 			detected = True
	# 			print("TARGET FOUND !!!")
	# 			roi = frame[y:y+h, x:x+w]
	# 			cv.imwrite("./Images/FOUND-"+str(endtime)+"_"+str(val)+".jpg", roi)
	# 		elif not detected and feed is True:
	# 			detected = True
	# 			smallsign = sign.createSign(signature,100)
	# 			small = sign.format_sign(smallsign)
	# 			big = sign.format_sign(bigsign)
	# 			smallList,bigList = get_similar_signs(smallsign,bigsign)
	# 			if len(smallList)+len(bigList) <= 5:
	# 				dynamic_data_entry(small,big)




	# show the frame and record if the user presses a key
	cv.imshow("Frame", frame)
	cv.imshow("Orig", orig)
	cv.imshow("GMASK", fgmask)


	endtime = endtime + timer()

	key = cv.waitKey(1) & 0xFF
 
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		createJsonLog(signList)
		break
	elif key != 255:
		cv.imwrite("./Images/"+chr(key)+"-frame.jpg" ,frame)
		#cv.imwrite("./Images/"+chr(key)+"-FGmask.jpg" ,fgmask)
		cv.imwrite("./Images/"+chr(key)+"-signature.jpg" , sign.GetSignature(frame,fgmask,x,y,w,h,True))


 
curs.close()
conn.close()
# cleanup the camera and close any open windows
camera.release()
cv.destroyAllWindows()