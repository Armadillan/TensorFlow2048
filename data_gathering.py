#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, re
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchWindowException, WebDriverException, NoSuchElementException

value_regex = re.compile(r"tile tile-(\d).*")

browser = webdriver.Firefox()
browser.get('https://play2048.co/')
html = browser.find_element_by_tag_name('html')
retry = browser.find_element_by_class_name('retry-button')
try:
    button = browser.find_element_by_class_name('ezmob-footer-close')
    while not button.is_displayed():
        pass
    time.sleep(.1)
    button.click()
    # button = browser.find_element_by_class_name('cookie-notice-dismiss-button')
    # while not button.is_displayed():
    #     pass
    # button.click()
except NoSuchElementException:
    pass

a_list = []
times_list = []

new_tiles = []

try:
    while True:

        start_time = time.time()

        html.send_keys(Keys.UP)
        html.send_keys(Keys.RIGHT)
        html.send_keys(Keys.DOWN)
        html.send_keys(Keys.LEFT)

        time.sleep(.1)

        if retry.is_displayed():
            a_list.append(int(browser.find_element_by_class_name('score-container').text))
            retry.click()

        if browser.find_elements_by_xpath('//div[@class="tile-inner" and contains(text(), "2048")]'):
            print('2048')
            break

        times_list.append(time.time() - start_time)

        tile = browser.find_element_by_class_name('tile-new')
        new_tiles.append(tile.get_attribute("class"))


except (NoSuchWindowException, WebDriverException):
    two = 0
    four = 0
    for i in new_tiles:
        if value_regex.match(i).group(1) == "2":
            two += 1
        else:
            four += 1
    fraction = four / (two + four)
    print(f"Twos: {two}\nFours: {four}\nFraction of fours: {fraction}\n")

    try:
        print('Number of games played: %s\nAverage core: %s\nHighscore: %s' % (len(a_list), sum(a_list)/len(a_list), max(a_list)))
        print('Average move time: %s seconds.' % (round(sum(times_list)/len(times_list), 7)))
    except ZeroDivisionError:
        print('No games finished')

