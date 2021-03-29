class Crawler():

    downloadPath = "./face_detecting_data/crawler/imageset"
    
    def __init__(self):
        self.count = 1
        self.driver = self.creatDriver()

    def setURL(self, url):
        self.url = url
        self.startCrawler()
    
    def creatDriver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('log-level=3')
        options.add_argument("disable-gpu")
        driver = webdriver.Chrome('./chromedriver.exe', chrome_options=options)
        driver.set_window_size(1920, 1080)
        return driver
    
    def dirCheck(self):
        if not os.path.isdir(self.downloadPath):                                                           
            os.mkdir(self.downloadPath)

    def scrollDown(self, driver):
        while(True):
            try:
                soup = self.getHTML(self.driver)
                more = soup.find(class_="YstHxe")
                end = soup.find(class_="DwpMZe")
                if (str(more).find("display:none;") == -1):
                    if(str(end).find('data-status="3"') == -1):
                        self.driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
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

        url_list = []

        for img in imgs:
            try:
                img.click()
                time.sleep(3)
                
                src = self.driver.find_element_by_xpath(
                '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img')
                src = src.get_attribute('src')
                url_list.append(src)
                urllib.request.urlretrieve(src, self.downloadPath + format(self.count, '04')+".jpg")
                self.count += 1
            except:
                pass

    def filtering(self):
        filtered_count = 0
        dir_name = "./download/"
        for index, file_name in enumerate(os.listdir(dir_name)):
            try:
                file_path = os.path.join(dir_name, file_name)
                img = Image.open(file_path)

                if img.width < 1280 and img.height < 720:
                    img.close()
                    os.remove(file_path)
                    filtered_count += 1

            except OSError:
                os.remove(file_path)
                filtered_count += 1

    def startCrawler(self):
        self.driver.get(self.url)
        self.scrollDown(self.driver)
        soup = self.getHTML(self.driver)
        imgs = self.getImage(soup)
        self.dirCheck()
        self.imgDownload(imgs)
        # self.filtering()

if __name__=='__main__':
    # print("URL입력")
    # url = input()
    crawler = Crawler()
    
    urls = [
                "https://www.google.com/search?q=%EB%B9%8C%EB%94%A9%20%ED%95%98%EB%8A%98&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCKC47aeUmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=%EB%8F%84%EC%8B%9C%20%ED%95%98%EB%8A%98&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCIingNaWmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=%EC%82%B0%20%ED%95%98%EB%8A%98&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCMC9sv2WmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=%EC%88%B2%20%ED%95%98%EB%8A%98&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCPD_zeOUmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937#imgrc=fHI-4ASdG81cwM",
                "https://www.google.com/search?q=%ED%98%B8%EC%88%98%20%ED%95%98%EB%8A%98&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCKjD8tqWmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=%EB%B0%94%EB%8B%A4%20%ED%95%98%EB%8A%98&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCPCyut6WmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=%ED%95%B4%EB%B3%80%20%ED%95%98%EB%8A%98&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCNjVkOmWmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=%ED%95%B4%EB%B3%80&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCLDWjfCWmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=sky%20mountain&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCKCtiYmXmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=lake%20sky&tbm=isch&hl=ko&tbs=isz:l&sa=X&ved=0CAEQpwVqFwoTCMj9spaXmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=city%20sky&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCJCg4ZyXmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=sky%20building&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCMCz48-XmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=%EB%A7%88%EC%9D%84%20%ED%95%98%EB%8A%98&tbm=isch&hl=ko&tbs=isz:l&sa=X&ved=0CAEQpwVqFwoTCIi6_O2XmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
                "https://www.google.com/search?q=village%20sky&tbm=isch&tbs=isz:l&hl=ko&sa=X&ved=0CAEQpwVqFwoTCKCywIGYmO4CFQAAAAAdAAAAABAC&biw=1903&bih=937",
        ]
    for url in tqdm(urls): 
        crawler.setURL(url)
    