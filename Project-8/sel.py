from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()


driver.get("https://www.google.com")


search_box = driver.find_element("name", "q")


search_box.send_keys("Selenium in Windows")

search_box.send_keys(Keys.RETURN)


driver.quit()