from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from time import sleep


class HabrPost:
    def __init__(self, post_id: int) -> None:
        response = requests.get(f"https://habr.com/ru/articles/{post_id}/")
        self.post_id = post_id
        if response.status_code == 200:
            self.soup = BeautifulSoup(response.text, features="html5lib")
        elif response.status_code in [404, 403]:
            self.soup = None
        elif response.status_code == 503:
            sleep(0.1)
            self.__init__(post_id)
        else:
            print(response.status_code)

    @property
    def author(self):
        if self.soup:
            return self.soup.find("a", "tm-user-info__username").text.strip()
    
    @property
    def title(self):
        if self.soup:
            title = self.soup.find("h1")
            if title:
                return title.text
        
    @property
    def tags(self):
        if self.soup:
            return list(map(lambda x: x.text.strip(" *"),
                            self.soup.find_all("a", "tm-publication-hub__link")))
            
    @property
    def text(self):
        if self.soup:
            raw_text = self.soup.find("div", "tm-article-body")
            if raw_text is None:
                raw_text = self.soup.find("div", "tm-post-snippet__content")
            return raw_text.text.strip()
    
    def as_dict(self):
        if self.soup:
            return {"post_id": self.post_id,
                     "author": self.author, 
                     "title": self.title, 
                     "tags": self.tags, 
                     "text": self.text}
        

class HabrParser:
    
    @staticmethod
    def parse_posts(n_posts: int, start: int = 807711, n_threads: int = 0, n_process: int = 0):

        if n_threads == n_process == 0:
            return HabrParser.parse_posts_single_process(n_posts=n_posts, start=start)
        
        elif n_threads > 0 and n_process == 0:
            return HabrParser.parse_posts_multithread(n_posts=n_posts, start=start, n_threads=n_threads)
        
        elif n_threads == 0 and n_process > 0:
            return HabrParser.parse_posts_multiprocess(n_posts=n_posts, start=start, n_process=n_process)
        
        else:
            raise Exception("Use only n_threads or the n_process, not together")

    @staticmethod
    def parse_posts_single_process(n_posts: int, start):
        res = []
        for i in tqdm(range(start, start-n_posts, -1)):
            post_dict = HabrParser.parse_single_post(i)
            if post_dict:
                res.append(post_dict)

        return res


    @staticmethod
    def parse_posts_multiprocess(n_posts: int, start:int, n_process: int = 3):

        with ProcessPoolExecutor(n_process) as pool:

            results = list(tqdm(pool.map(HabrParser.parse_single_post,
                                         range(start, start-n_posts, -1)), 
                                         total=n_posts))
        
        return [i for i in results if i]

    @staticmethod
    def parse_posts_multithread(n_posts: int, start:int, n_threads: int = 3):

        with ThreadPoolExecutor(n_threads) as pool:

            results = list(tqdm(pool.map(HabrParser.parse_single_post,
                                         range(start, start-n_posts, -1)), 
                                         total=n_posts))
        
        return [i for i in results if i]
    
    @staticmethod
    def parse_single_post(post_id):
        try:
            post_dict = HabrPost(post_id).as_dict()
            return post_dict

        except Exception as e:
            print(f"Произошла ошибка для id = {post_id}", e)
