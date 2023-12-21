#   Author: Besker Telisma
#   Creation Date: 2023-12-07 14:40:46.180138
#   Automation for:
#        com.hulu.plus




from appium import webdriver
import time
from appium.webdriver.common.touch_action import TouchAction
import json
from selenium.common.exceptions import ScreenshotException
import sys


# update important information into a json file, do not delete.
runData_obj = {"script_type": "appium", "platform": "Android", "login_success": False, "actions_completed": 0, "total_actions": 5, "os_version": ""}

def launchSession():
    caps = {"deviceName": "Android",
            "platformName": "Android",
            "appPackage": "com.hulu.plus",
            "appActivity": "com.hulu.features.splash.SplashActivity",
            "automationName": "UiAutomator2",
            "ensureWebviewsHavePages": True,
            'noSign': True}

    return(webdriver.Remote("http://localhost:4723/wd/hub", caps))

#  update important information into a json file, do not delete.
def collectDeviceData(driver):
    try:
        runData_obj["os_version"] = driver.capabilities["platformVersion"]
    except Exception as err:
        driver.log_event("nowsecure", f"Device Data Collection Error: {err=}, {type(err)=}")

def tapByCoordinates(driver, x1, y1):
    targetDevice = driver.capabilities["deviceModel"][-2:]
    localDevice = "3a"

    screenHeights = {
        "3a": 2080,
        "4a": 2255,
        "5a": 2340
    }

    if localDevice != targetDevice:
        print("targetDevice != localDevice; if you are testing locally, update tapByCoordinates so that localDevice is your Pixel model (3a, 4a or 5a)")
        y2 = round(y1 * (screenHeights[targetDevice] / screenHeights[localDevice]))
    else:
        y2 = y1

    print(f"y1 = {y1}")
    print(f"y2 = {y2}")

    logString = "Clicking by coordinates: x=" + str(xCoord) + " y=" + str(yCoord)
    driver.log_event("nowsecure", logString)
    tapAction = TouchAction(driver)
    tapAction.tap(x=x1, y=y2).perform()
    runData_obj["actions_completed"] += 1


# scroll function
def scroll(driver, fromX, fromY, toX, toY):
    runData_obj["actions_completed"] += 1
    driver.swipe(fromX, fromY, toX, toY)

# it scroll the left side of the screen
def big_scroll(driver):
    #FIXME: Besker to update this function to scroll to the bottom of the page when invoked.
    runData_obj["actions_completed"] += 1
    driver.swipe(160, 1570, 150, 250)
    

# get rid of android permissions
def permission(driver):
    # NOTE: permission actions are not expected to always fire, so total_actions should be incremented if and only if a click happens
    runData_obj["total_actions"] += 1
    if waitExist(driver, "//android.widget.Button[@text='Allow']", "xpath", 1):
        waitFor(driver, "//android.widget.Button[@text='Allow']", "xpath", 1)[0].click()
    elif waitExist(driver, "com.android.permissioncontroller:id/permission_allow_one_time_button", "id", 1):
        waitFor(driver, "com.android.permissioncontroller:id/permission_allow_one_time_button", "id", 1)[0].click()
    elif waitExist(driver, "com.android.permissioncontroller:id/permission_allow_button", "id", 1):
        waitFor(driver, "com.android.permissioncontroller:id/permission_allow_button", "id", 1)[0].click()
    elif waitExist(driver, "com.android.permissioncontroller:id/permission_allow_foreground_only_button", "id", 1):
        waitFor(driver, "com.android.permissioncontroller:id/permission_allow_foreground_only_button", "id", 1)[0].click()
    elif waitExist(driver, "//android.widget.Button[@text='Allow Once']", "xpath", 1):
        waitFor(driver, "//android.widget.Button[@text='Allow Once']", "xpath", 1)[0].click()
    elif waitExist(driver, "//android.widget.Button[@text='Always Allow']", "xpath", 1):
        waitFor(driver, "//android.widget.Button[@text='Always Allow']", "xpath", 1)[0].click()
    elif waitExist(driver, "//android.widget.Button[@text='OK']", "xpath", 1):
        waitFor(driver, "//android.widget.Button[@text='OK']", "xpath", 1)[0].click()
    else:
        runData_obj["total_actions"] -= 1

def handleChrome(driver):
    if waitExist(driver, "com.android.chrome:id/signin_fre_dismiss_button", "id"):
        runData_obj["total_actions"] += 1
        waitFor(driver, "com.android.chrome:id/signin_fre_dismiss_button", "id")[0].click()
    if waitExist(driver, 'com.android.chrome:id/terms_accept', 'id', 5):
        runData_obj["total_actions"] += 1
        waitFor(driver, 'com.android.chrome:id/terms_accept', 'id', 1)[0].click()
        if waitExist(driver, 'com.android.chrome:id/negative_button', 'id', 5):
            runData_obj["total_actions"] += 1
            waitFor(driver, 'com.android.chrome:id/negative_button', 'id', 1)[0].click()

def waitFor(driver, elString, pathType, timeoutMax=30):
    takeScreenshot(driver)
    runData_obj["actions_completed"] += 1
    print("Searching for " + elString + " by " + pathType)
    timeoutVariable = 0
    while timeoutVariable < timeoutMax:

        if pathType == 'id':
            elx = driver.find_elements_by_id(elString)
        elif pathType == 'xpath':
            elx = driver.find_elements_by_xpath(elString)
        elif pathType == 'accessibility_id':
            elx = driver.find_elements_by_accessibility_id(elString)
        else:
            print("pathType Not Supported")
            exit()

        if len(elx) > 0:
            return(elx)

        else:
            print("waiting...")

        timeoutVariable += 1
        time.sleep(1)

    raise Exception("Element not present")
    exit()

def waitExist(driver, elString, pathType, timeoutMax=30):
    takeScreenshot(driver)
    print("Searching for " + elString + " by " + pathType)
    timeoutVariable = 0
    while timeoutVariable < timeoutMax:
        try:
            if pathType == 'id':
                elx = driver.find_element_by_id(elString)
                return(True)
            elif pathType == 'xpath':
                elx = driver.find_element_by_xpath(elString)
                return(True)
            elif pathType == 'accessibility_id':
                elx = driver.find_element_by_accessibility_id(elString)
                return(True)
            else:
                print("pathType Not Supported")
                exit()

        except:
            print("waiting...")

        timeoutVariable += 1
        time.sleep(1)
    return(False)

def takeScreenshot(driver):
    try:
        driver.get_screenshot_as_base64()
    except ScreenshotException:
        print("Could not take screenshot; FLAG_SECURE enabled")
        driver.log_event("nowsecure", "Could not take screenshot; FLAG_SECURE enabled")

# record login success
def success(driver, elString, pathType, timeoutMax=30):
    if waitExist(driver, elString, pathType, timeoutMax):
        runData_obj["login_success"] = True

#import configuration data file which will be copied automatically to the appium folder in platfrom
def importConfig(driver):
    try:
        f = open('config.json')

        # returns JSON object as
        # a dictionary
        return(json.load(f))

    except Exception as err:
        print("Failed to open json file")
        print(f"Unexpected {err=}, {type(err)=}")
        driver.log_event("nowsecure", "Failed to open json file")
        driver.log_event("nowsecure", f"Unexpected {err=}, {type(err)=}")
        sys.exit()

def primaryFunctionality(driver):
    #import config data
    data = importConfig(driver)

    # Username and password will be pulled from the config file in platform
    # This will allow us to dynamically adjust credentials without needing a rescript
    #username & password can we used as variables.
    try:
        username = data["search_data"]["username"]["value"]
        password = data["search_data"]["password"]["value"]
        driver.log_event("nowsecure", username)
        driver.log_event("nowsecure", password)

    #if the creds cannot be ingested then the assessment will exit.
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        driver.log_event("nowsecure", f"Unexpected {err=}, {type(err)=}")
        sys.exit()


    
    # username = 'f-dmed-appsec-mast-a@disney.com'
    # password ='GF2MGEqfXY2B^tOc0x&Eh'

    waitFor(driver,"//android.widget.Button[@text='LOG IN']", "xpath", 30)[0].click()

    time.sleep(10)
    waitFor(driver,"//android.widget.EditText","xpath",30)[0].send_keys(username)
    waitFor(driver,"//android.widget.EditText","xpath",30)[1].send_keys(password)

    waitFor(driver,"//android.widget.Button[@text='LOG IN']", "xpath", 30)[0].click()


    if waitExist(driver,"//*[contains(@text,'GOT IT')]", "xpath",30):
        waitFor(driver,"//*[contains(@text,'GOT IT')]", "xpath",30)[0].click()

    # add the total actions expected to the rundata dictionary
    # to record login success enter the element expected as the second argument, ex: success(driver, "//android.widget.Button[@text='OK']")
    # or success(driver, "com.android.chrome:id/terms_accept")
    success(driver,"//*[contains(@text,'HOME')]", "xpath", 30)
    takeScreenshot(driver)



if __name__ == "__main__":
    driver = launchSession()
    driver.log_event("nowsecure", "Session Launched")
    driver.log_event("nowsecure", "RunData Present in Script")

    try:
        primaryFunctionality(driver)
    except Exception as err:
        print(f"Workflow exited with {err=}, {type(err)=}")
        driver.log_event("nowsecure", f"Workflow exited with {err=}, {type(err)=}")

    collectDeviceData(driver)
    driver.log_event("nowsecure", f"RunData:{json.dumps(runData_obj)}")
    print('appium', f"RunData:{json.dumps(runData_obj)}")
    takeScreenshot(driver)
    print("Session Complete")
    driver.quit()
