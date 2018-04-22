from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import sys
import csv
import time

# Dicts replacing if statements between user and critic reviews
REVIEW_TYPES_URL = {'1' : "user-reviews", '2' : "critic-reviews"}
REVIEWS_TYPES_TAG = {'1' : "review user_review", '2' : "review critic_review"}
ARRAY_INDICES_SCORE = {'1' : 2, '2' : 1}
ARRAY_INDICES_REVIEW = {'1' : 3, '2' : 2}

# Taking user input
game = input("Game: ")
plat = input("Platform(pc, xbox-one, playstation-4): ")
reviewType = input("'1' for USER REVIEWS or '2' for CRITIC REVIEWS: ")

# Name of file that will contain the data
fName = "new/" + game + "-" + plat + "-" + REVIEWS_TYPES_TAG[reviewType] + "-" + time.strftime("%Y-%m-%d %H:%M") + ".csv"

# Opening the csv file
file = open(fName, 'w')

# Current page the script is on, starts at 0
pageCounter = 0

# Last page for the current set of reviews being processed, starts at max int
lastPage = sys.maxsize

# Loop to get all pages, each iteration is a page
while (pageCounter < lastPage):

	# Go to next page
	pageCounter += 1

	# Making the url for the current page
	url = "http://www.metacritic.com/game/" + plat + "/" + game + "/" + REVIEW_TYPES_URL[reviewType] + "?page=" + str(pageCounter)	

	# Request for the current page
	req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
	web_byte = urlopen(req).read()
	page = web_byte.decode('utf-8')
	soup = BeautifulSoup(page, 'html.parser')

	# Getting the reviews
	# Issue of repeating what comes before the 'expand' word in long reviews
	reviews = soup.findAll('li', attrs={'class': REVIEWS_TYPES_TAG[reviewType]})

	# Parsing the reviews, each iteration is a review
	for review in reviews:
		# Getting the username
		user = review.find('div', attrs={'class': 'name'}).text.strip()
		# Getting the date
		date = review.find('div', attrs={'class': 'date'}).text.strip()
		# Getting the score
		score = review.find('div', attrs={'class': 'review_grade'}).text.strip()
		# Unscored reviews in the critics section
		if not score:
			score = "IN PROGRESS & UNSCORED"

		reviewBody = review.find('div', attrs={'class': 'review_body'}).text
		reviewBody = reviewBody.replace("\r", "").replace("\n", "").replace("… Expand", "").replace("Read full review", "")
		time.sleep(0.06)
		# Writing to the csv file
		file.write(date + "," + user + "," + score + "," + reviewBody + "\n")
			# replace("… Expand", "").replace(",", "").replace("\n", "").replace("\r", "").replace("Read full review", "").strip() + ",\n")

	# Getting last page and comparing it to current, could be factored out
	try:
		lastPage = soup.find('li', attrs={'class': 'page last_page'}).text
		lastPage = lastPage.replace("…", "")
		lastPage = int(lastPage)
	except AttributeError:
		lastPage = 0

# Closing file
file.close()