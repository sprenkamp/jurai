import os
import logging
import time
from natsort import natsorted
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Configure logging
logging.basicConfig(filename='scraping_log.txt', level=logging.DEBUG)

# Function to accept cookies
def accept_cookies(driver):
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#CybotCookiebotDialogBodyContent"))
        )

        accept_cookie_button = driver.find_element(By.CSS_SELECTOR, "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowallSelection")
        accept_cookie_button.click()

        logging.info("Cookies accepted.")

    except TimeoutException:
        logging.info("Cookie banner not found or already accepted.")

# Function to perform login
def login(driver, username, password):
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "kennung")))

        driver.execute_script('document.getElementById("kennung").value = arguments[0];', username)
        driver.execute_script('document.getElementById("password").value = arguments[0];', password)
        driver.execute_script('document.getElementById("loginbutton").click();')

        print("Login successful.")

    except Exception as e:
        print(f"Error during login: {e}")

# Function to wait for an element to be clickable
def wait_for_clickable(driver, by, value, timeout=10):
    return WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((by, value)))

# Function to download RTF file
def download_rtf_file(driver, output_dir, output_filename):
    try:
        # Click on the RTF icon to download the file
        rtf_icon = wait_for_clickable(driver, By.XPATH, '//*[@id="docBar"]/ul/li[2]/a/div')
        logging.info("Clicking on RTF icon.")
        rtf_icon.click()

        # Wait for the file to be downloaded (you may need to adjust the sleep duration)
         # Use explicit wait for the file to be downloaded
        wait = WebDriverWait(driver, 10)
        wait.until(lambda driver: os.path.isfile(os.path.join(output_dir, "download.rtf")))  # Increase the implicit wait to ensure file download completion
        # You can add additional checks for file download completion if necessary

        # Get the downloaded file name
        downloaded_file_name = "download.rtf"  # Adjust this based on the actual downloaded file name

        # Move the downloaded file to the output directory
        downloaded_file_path = os.path.join(output_dir, downloaded_file_name)
        os.rename(downloaded_file_name, downloaded_file_path)

        logging.info(f"RTF file downloaded and saved: {downloaded_file_path}")

    except Exception as e:
        logging.error(f"Error downloading RTF file: {e}")

# Main function
def main():
    input_dir = "/Users/therealchrisbrennan/Documents/Project/JurAI/All links"
    output_dir = "/Users/therealchrisbrennan/Documents/Project/JurAI/Entscheidungen fertig"
    processed_links_file = os.path.join(output_dir, "processed_links.txt")

    # Get list of links from text files in chronological order
    links = []
    for file in natsorted(os.listdir(input_dir)):  # natsorted for natural sorting
        if file.startswith("Link_") and file.endswith(".txt"):
            with open(os.path.join(input_dir, file), 'r', encoding='utf-8') as text_file:
                link = text_file.read().strip()
                links.append((file, link))

    username = "christopher.brennan"
    password = "venivedivici"

    # Use a counter that increments by 1
    link_count = 1

    # Create processed links file in the output directory if it doesn't exist
    with open(processed_links_file, 'a'):  # 'a' mode for appending or creating if not exists
        pass

    for input_file, link in links:
        # Check if link has been processed before
        with open(processed_links_file, 'r') as processed_links:
            if link in processed_links.read():
                logging.info(f"Link {link_count} already processed. Skipping...")
                continue

        # Set up Chrome options for headless mode
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")

        # Set download path for Chrome
        prefs = {"download.default_directory": output_dir}
        chrome_options.add_experimental_option("prefs", prefs)

        # Create a WebDriver instance using 'with'
        with webdriver.Chrome(options=chrome_options) as driver:

            try:
                print(f"Processing link from file: {input_file}")
                driver.get(link)  # Open the URL

                accept_cookies(driver)
                login(driver, username, password)

                # Download RTF file
                print(f"Downloading RTF file for {link}")
                download_rtf_file(driver, output_dir, f"link_{link_count}.rtf")

                # Append the processed link to the processed links file
                with open(processed_links_file, 'a') as processed_links:
                    processed_links.write(f"{input_file}: {link}\n")

                link_count += 1

            except Exception as e:
                logging.error(f"Error processing link {link_count}: {e}")

            finally:
                # Close the WebDriver instance
                driver.quit()

if __name__ == "__main__":
    main()