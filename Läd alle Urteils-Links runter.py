import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def accept_cookies(driver):
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#CybotCookiebotDialogBodyContent"))
        )

        accept_cookie_button = driver.find_element(By.CSS_SELECTOR, "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowallSelection")
        accept_cookie_button.click()

        print("Cookies accepted.")

    except TimeoutException:
        print("Cookie banner not found or already accepted.")

def click_error_confirmation_ok_button(driver):
    try:
        error_confirmation_ok_button = driver.find_element(By.XPATH, '//*[@id="error_confirmation_ok_button"]')
        error_confirmation_ok_button.click()

        print("Clicked on error confirmation OK button.")

    except Exception as e:
        print(f"Error clicking error confirmation OK button: {e}")

def login(driver, username, password):
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "kennung")))

        driver.execute_script('document.getElementById("kennung").value = arguments[0];', username)
        driver.execute_script('document.getElementById("password").value = arguments[0];', password)
        driver.execute_script('document.getElementById("loginbutton").click();')

        print("Login successful.")

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".documentHeaderCol:nth-child(2) > table"))
        )

    except Exception as e:
        print(f"Error during login: {e}")

def click_categories_select_Vorschriften(driver):
    try:
        click_categories_select_Vorschriften = driver.find_element(By.XPATH, '//*[@id="categories_select_Vorschriften"]/span[1]')
        click_categories_select_Vorschriften.click()

        print("Clicked on categories_select_Vorschriften.")

    except Exception as e:
        print(f"Error clicking categories_select_Vorschriften: {e}")

def click_search_bar_search(driver):
    try:
        search_bar_search = driver.find_element(By.XPATH, '//*[@id="searchBar_search"]/em')
        search_bar_search.click()

        print("Clicked on search bar search.")

        # Wait until all links are loaded
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//a[contains(@href, "document") or contains(@href, "pdf")]')))

    except Exception as e:
        print(f"Error clicking search bar search: {e}")

def scroll_down(driver):
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            WebDriverWait(driver, 10).until(lambda driver: driver.execute_script("return document.readyState") == "complete")

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break

            last_height = new_height

    except Exception as e:
        print(f"Error during scrolling: {e}")

def scrape_and_save_links(driver, output_dir, processed_links):
    try:
        page = 1
        while True:
            # Find all links for a document on the page
            links = driver.find_elements(By.XPATH, '//a[contains(@href, "document") or contains(@href, "pdf")]')

            # Save links for the document in the output directory
            for idx, link in enumerate(links, start=(page-1)*len(links)+1):
                link_href = link.get_attribute('href')

                # Check if the link has already been processed and saved
                if link_href not in processed_links:
                    filename = f"{output_dir}/Link_{idx}.txt"
                    with open(filename, 'w', encoding='utf-8') as file:
                        file.write(link_href)

                    print(f"Link {idx} saved: {filename}")
                    processed_links.add(link_href)

            # Check if there is a "Next" button
            next_button = driver.find_elements(By.CLASS_NAME, 'svg-icon-chevron_right')
            if next_button:
                # Click the "Next" button to navigate to the next page
                next_button[0].click()

                # Wait for a short duration to allow content to load on the next page
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//a[contains(@href, "document") or contains(@href, "pdf")]')))
                WebDriverWait(driver, 10).until(lambda driver: driver.execute_script("return document.readyState") == "complete")

                # Introduce a delay to avoid overloading the server
                time.sleep(2)
            else:
                break  # Exit the loop if there is no "Next" button, indicating the end of pagination

            page += 1

    except Exception as e:
        print(f"Error during scraping and saving links: {e}")

def main(headless=True):
    url = "https://www.juris.de/jportal/nav/index.jsp"
    output_dir = "/Users/therealchrisbrennan/Documents/Project/JurAI/Gesetze"  # Adjust the output directory

    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')

    driver = webdriver.Chrome(options=options)
    processed_links = set()  # To keep track of processed links

    # Open the URL
    driver.get(url)

    # Accept cookies
    accept_cookies(driver)

    # Click on error confirmation OK button
    click_error_confirmation_ok_button(driver)

    # Perform login after opening the link
    login(driver, "christopher.brennan", "venivedivici")

    # Click on categories_select_Rechtsprechung
    click_categories_select_Vorschriften(driver)

    # Click on search bar search
    click_search_bar_search(driver)

    # Scrape and save links with pagination
    scrape_and_save_links(driver, output_dir, processed_links)

    # Close the WebDriver
    driver.quit()

if __name__ == "__main__":
    main()
