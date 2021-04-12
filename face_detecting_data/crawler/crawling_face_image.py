
import cv2
import time
import os
import urllib.request
import sys
sys.path.append(r'C:\Users\Toz\Desktop\pytorch\term_project')
from face_detecting_data.face_detect import getFacePosition
from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

class Crawler():

    downloadPath = "./face_detecting_data/crawler/imageset/"
    
    def __init__(self):
        self.count = 1
        self.driver = self.creatDriver()

    def setURL(self, emotion, url):
        self.emotion = emotion
        self.url = url
        self.startCrawler()
        
    
    def creatDriver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('log-level=3')
        options.add_argument("disable-gpu")
        driver = webdriver.Chrome('./face_detecting_data/crawler/chromedriver.exe', chrome_options=options)
        driver.set_window_size(1920, 1080)
        return driver
    
    def dirCheck(self):
        if not os.path.isdir(self.downloadPath):                                                           
            os.mkdir(self.downloadPath)

        if not os.path.isdir(self.downloadPath+"Fear"):                                                           
            os.mkdir(self.downloadPath+"Fear")

        if not os.path.isdir(self.downloadPath+"Sadness"):                                                           
            os.mkdir(self.downloadPath+"Sadness")

        if not os.path.isdir(self.downloadPath+"Joy"):                                                           
            os.mkdir(self.downloadPath+"Joy")
            
        if not os.path.isdir(self.downloadPath+"Disgust"):                                                           
            os.mkdir(self.downloadPath+"Disgust")

        if not os.path.isdir(self.downloadPath+"Anger"):                                                           
            os.mkdir(self.downloadPath+"Anger")

    def scrollDown(self, driver):
        while(True):
            try:
                soup = self.getHTML(self.driver)
                more = soup.find(class_="YstHxe")
                accdown = soup.find(class_="WYR1I")
                end = soup.find(class_="DwpMZe")
                if (str(more).find("display:none;") == -1):
                    if(str(end).find('data-status="3"') == -1):
                        
                        self.driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[4]/div[2]/input').click()
                    else:
                        break
                self.driver.execute_script("window.scrollBy(0,10000)")
                time.sleep(1)
                self.driver.execute_script("window.scrollBy(0,10000)")
                time.sleep(1)
            except:
                self.driver.execute_script("window.scrollBy(0,10000)")
                time.sleep(1)
                pass

    def getHTML(self, driver):
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        return soup

    def getImage(self, soup):
        img_src = self.driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
        return img_src

    def imgDownload(self, imgs):

        # url_list = []

        for img in imgs:
            try:
                img.click()
                time.sleep(3)
                src = self.driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img')
                src = src.get_attribute('src')
                # url_list.append(src)
                urllib.request.urlretrieve(src, self.downloadPath + self.emotion + f"/{self.emotion}__{format(self.count, '04')}.jpg")
                self.count += 1
            except Exception as e:
                print(e)
                pass

    def filtering(self):
        filtered_count = 0
        dir_name = f"./face_detecting_data/crawler/imageset/{self.emotion}/"
        count = 0
        dir_list = os.listdir(dir_name)
        for index, file_name in enumerate(dir_list):
            file_path = os.path.join(dir_name, file_name)
            img = cv2.imread(file_path)
            
            faces = getFacePosition(img)
            if len(faces) == 0:
                os.remove(file_path)
            else:
                for face in faces:
                    try:
                        face_img = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
                        cv2.imwrite(f"{dir_name}{self.emotion}_{format(count, '04')}.png", face_img)
                        os.remove(file_path)
                        count += 1
                    except Exception as e:
                        print(e)
            


    def startCrawler(self):
        self.driver.get(self.url)
        self.scrollDown(self.driver)
        soup = self.getHTML(self.driver)
        imgs = self.getImage(soup)
        self.dirCheck()
        self.imgDownload(imgs)
        self.filtering()

if __name__=='__main__':
    # print("URL입력")
    # url = input()
    crawler = Crawler()
    
    urls ={
        'Anger' : 'https://www.google.com/search?q=Anger+face&tbm=isch&chips=q:happiness+face,online_chips:human&hl=ko&sa=X&ved=2ahUKEwiji4T96NTvAhXCEXAKHZKxBh4Q4lYoAnoECAEQGw&biw=1029&bih=845',
        'Disgust' : 'https://www.google.com/search?q=Disgust+face&tbm=isch&chips=q:happiness+face,online_chips:human&hl=ko&sa=X&ved=2ahUKEwiji4T96NTvAhXCEXAKHZKxBh4Q4lYoAnoECAEQGw&biw=1029&bih=845',
        'Joy' : 'https://www.google.com/search?q=happiness+face&tbm=isch&chips=q:happiness+face,online_chips:human&hl=ko&sa=X&ved=2ahUKEwiji4T96NTvAhXCEXAKHZKxBh4Q4lYoAnoECAEQGw&biw=1029&bih=845',
        'Sadness' : 'https://www.google.com/search?q=Sadness+face&tbm=isch&chips=q:happiness+face,online_chips:human&hl=ko&sa=X&ved=2ahUKEwiji4T96NTvAhXCEXAKHZKxBh4Q4lYoAnoECAEQGw&biw=1029&bih=845',        
        'Fear' : 'https://www.google.com/search?q=fear+face&tbm=isch&chips=q:fear+face,online_chips:human&hl=ko&sa=X&ved=2ahUKEwicgtek6dTvAhVRfHAKHRcXB8AQ4lYoAnoECAEQGw&biw=1029&bih=845',
        }

    for key in tqdm(urls): 
        crawler.setURL(key, urls[key])
    