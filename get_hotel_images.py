# Written by Mary Wahl for a //oneweek tutorial
#
# Based heavily on atif93's response to a related question on StackOverflow:
# https://stackoverflow.com/questions/35809554/
#      how-to-download-google-image-search-results-in-python

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os, json, urllib, shutil, time, argparse

def main(search_text, n_images, download_path):
    search_text = '+'.join(search_text.strip().split(' '))
    n_scrolls = int(n_images / 400) + 1

    url = 'https://www.google.com/search?q=' + \
          '{}&source=lnms&tbm=isch'.format(search_text)
    driver = webdriver.Firefox()
    driver.get(url)

    headers = {}
    headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/' + \
                            '537.36 (KHTML, like Gecko) Chrome/41.0.2228.0' + \
                            ' Safari/537.36'
    extensions = {'jpg', 'jpeg', 'png', 'gif'}
    img_count = 0
    downloaded_img_count = 0

    for _ in range(n_scrolls):
        for __ in range(10):
            driver.execute_script('window.scrollBy(0, 1000000)')
            time.sleep(0.2)
        time.sleep(0.5)
        try:
            driver.find_element_by_xpath("//input[@value='Show more " + \
                                         "results']").click()
        except Exception as e:
            print('Fewer images found: {}'.format(e))
            break

    images = driver.find_elements_by_xpath("//div[@class='rg_meta notranslate']")
    print('Total images: {}'.format(len(images)))
    for img in images:
        img_count += 1
        img_url = json.loads(img.get_attribute('innerHTML'))['ou']
        img_type = json.loads(img.get_attribute('innerHTML'))['ity']
        try:
            if img_type not in extensions:
                continue
            req = urllib.request.Request(img_url, headers=headers)
            raw_img = urllib.request.urlopen(req).read()
            with open(os.path.join(download_path,
                                   '{}.{}'.format(downloaded_img_count,
                                                  img_type)),
                      'wb') as f:
                f.write(raw_img)
            downloaded_img_count += 1
        except Exception as e:
            print('Download failed: {}'.format(e))
        finally:
            print
        if downloaded_img_count >= n_images:
            break

    print('Total downloaded: {}/{}'.format(downloaded_img_count, img_count))
    driver.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Retrieves image files matching a search query on Google Images. Output directory
will be removed if it already exists.
''')
    parser.add_argument('-q', '--query', type=str, required=True,
                        help='Search string (enclosed in double quotes)')
    parser.add_argument('-n', '--n_images', type=int, required=True,
                        help='Max number of images to store')
    parser.add_argument('-o', '--output_dir',
                        type=str, required=True,
                        help='Output directory for images')
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    main(args.query, args.n_images, args.output_dir)