import sys

import urllib.request
from urllib.request import Request, urlopen
from urllib.parse import quote
import http.client

import time  # Importing the time library to check the time of code execution
import os
import argparse
import ssl
import datetime
import json
import re
import codecs
import socket

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from random import randint

def create_directories(main_directory, dir_name):
	print("We created the directories.")
	# make a search keyword  directory
	try:
		if not os.path.exists(main_directory):
			os.makedirs(main_directory)
			time.sleep(0.2)
			path = (dir_name)
			sub_directory = os.path.join(main_directory, path)
			if not os.path.exists(sub_directory):
				os.makedirs(sub_directory)
		else:
			path = (dir_name)
			sub_directory = os.path.join(main_directory, path)
			if not os.path.exists(sub_directory):
				os.makedirs(sub_directory)
	except OSError as e:
		if e.errno != 17:
			raise
		pass
	return

def build_search_url(search_term):
	url = 'https://www.google.com/search?q=' + quote(
		search_term.encode('utf-8')) + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'

	return url

def download_extended_page(url):
    if sys.version_info[0] < 3:
        reload(sys)
        sys.setdefaultencoding('utf8')

    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument("--headless")

    try:
        browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    except Exception as e:
        sys.exit()

    browser.set_window_size(1024, 768)

    # Open the link
    print(url)
    browser.get(url)
    time.sleep(1)

    print("Getting you a lot of images. This may take a few moments...")

    element = browser.find_element(by=By.TAG_NAME, value="body")
    # Scroll down
    for i in range(30):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    try:
        browser.browser.find_element(by=By.ID, value="smb").click()
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection
    except:
        for i in range(10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection

    print("Reached end of Page.")
    time.sleep(0.5)

    source = browser.page_source
    browser.close()

    return source

# Getting all links with the help of '_images_get_next_image'
def get_all_items(page, main_directory, dir_name, limit):
	
    i = 0
    count = 1

    urls = re.findall('\"https://encrypted.+?;usqp=CAU\"', page)
    urls = [url[1:-1] for url in urls]

    # f = open("urls.txt", "a")
    # f.write('\n'.join(urls))
    # f.close()

    while count < limit+1:
        #download the images
        url = urls[count]
        download_image(url, main_directory, dir_name, count)
        count += 1

def download_image(image_url, main_directory, dir_name, count):
    try:
        req = Request(image_url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
        try:
            # timeout time to download an image
            timeout = 10

            response = urlopen(req, None, timeout)
            data = response.read()
            response.close()

            path = main_directory + "/" + dir_name + "/" + str(randint(1, 100000000)) + ".jpg"

            try:
                output_file = open(path, 'wb')
                output_file.write(data)
                output_file.close()
                absolute_path = os.path.abspath(path)

            except:
                print("Error saving the image")

        except:
            print('Error')

    except:
        print('Error')
