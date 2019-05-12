import cv2
import numpy as np
import urllib.request
from bs4 import BeautifulSoup
import re


#resizing image to 256*256 and saving to "dir"
def save_image_by_url(url, dir, name):
    req = urllib.request.Request(url=url, headers={'User-Agent': 'Chrome'})
    img_array = np.asarray(bytearray(urllib.request.urlopen(req).read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    new_img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    cv2.imwrite(dir + '/' + name + '.jpg', new_img)


#source 1: pexels.com
def save_from_source_1(num_of_pages, animal_name):
    headers = {'User-Agent': 'Chrome'}

    for i in range(1, num_of_pages + 1):
        url = 'https://www.pexels.com/search/' + animal_name + '/?page=' + str(i)
        req = urllib.request.Request(url=url, headers=headers)
        page = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(page, 'html.parser')

        img_id = 0
        for img in soup.find_all('img'):
            src = img.get('src')
            if ('jpeg' in src):
                img_id += 1
                src = re.findall('[a-zA-Z_/:\-.0-9]+', src)
                if 'avatars' not in src[0]:
                    save_image_by_url(src[0], 'founded_images/' + animal_name, str(i) + '_' + str(img_id))

        print('page ' + str(i) + ' of ' + str(num_of_pages))


#source 2: pixabay.com
def save_from_source_2(num_of_pages, animal_name):
    headers = {'User-Agent': 'Chrome'}

    for i in range(1, num_of_pages + 1):
        url = 'https://pixabay.com/images/search/' + animal_name + '/?pagi=' + str(i)
        req = urllib.request.Request(url=url, headers=headers)
        page = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(page, 'html.parser')

        img_id = 0
        for img in soup.find_all('img'):
            src = img.get('src')
            if ('jpg' in src):
                img_id += 1
                save_image_by_url(src, 'founded_images/' + animal_name, 'second_' + str(i) + '_' + str(img_id))

        print('page ' + str(i) + ' of ' + str(num_of_pages))


#source 3: gettyimages.com
def save_from_source_3(num_of_pages, animal_name):
    headers = {'User-Agent': 'Chrome'}

    for i in range(1, num_of_pages + 1):
        url = 'https://www.gettyimages.com/photos/' + animal_name + '?page=' + str(i)

        req = urllib.request.Request(url=url, headers=headers)
        page = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(page, 'html.parser')

        img_id = 0
        for img in soup.find_all('img'):
            img_id += 1
            try:
                src = img['src']
                save_image_by_url(src, 'founded_images/' + animal_name, 'third_0' + str(i) + '_' + str(img_id))
            except:
                pass

        print('page ' + str(i) + ' of ' + str(num_of_pages))


if __name__ == '__main__':
    #use your functions here, for example:
    #save_from_source_3(num_of_pages=20, animal_name='owl')
    #function will save your images to ./founded_images/animal_name/
