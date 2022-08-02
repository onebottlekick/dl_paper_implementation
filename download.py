import os
import socket
from urllib.error import HTTPError
from urllib.request import urlretrieve

import mistune
from bs4 import BeautifulSoup
from tqdm import tqdm


def clean_title(title, targets={':': '_', ' ': '_', '/': '_', '.': '', '"': ''}):
    for k, v in targets.items():
        title = title.replace(k, v)
    return title


def get_download_info(source='README.md'):
    with open(source, encoding='utf-8') as f:
        f_html = mistune.markdown(f.read())
        f_soup = BeautifulSoup(f_html, 'html.parser')
    
    paths = [path.text.replace(' ', '-') for  path in f_soup.find_all('h2')]
    contents = f_soup.find_all('ul')
    contents = [{clean_title(k.text.replace(' [pdf]', '').replace(' [code]', '')):v.attrs['href'] for k, v in zip(content.find_all('li'), [c for c in content.find_all('a') if c.text != '[code]'])} for content in contents]
        
    return dict(zip(paths, contents))
        

def download(title, url, path):
    try:
        full_path = os.path.join(path, title)
        if not os.path.exists(full_path):            
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=title) as t:
                urlretrieve(url, full_path, reporthook=t.update_to)
        else:
            print(f'{full_path} already exists')
            
    except HTTPError:
        print('Error 404')
        raise 
    
    except socket.timeout:
        print('connection timeout')
        raise


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b*bsize - self.n)

if __name__ == '__main__':
    info = get_download_info()
    for path, content in info.items():
        path = os.path.join('papers', path)
        os.makedirs(path, exist_ok=True)
        for title, url in content.items():
            download(title + '.pdf', url, path)
