import requests

def get_platforms():
    base_url = 'http://www.giantbomb.com/api'
    api_key = 'a0ba8e3ae03028e28fdae2671699af48caf2aeb6'
    params = '&format=json&field_list=name&offset='
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'calebstrait',
    })
    req = base_url + '/platforms/?api_key=' + api_key + params + str(0)
    print('\nreq:' + req + '\n')
    result = requests.get(req, headers=headers)
    print(result.text)
    print('\n***\n')
    req = base_url + '/platforms/?api_key=' + api_key + params + str(100)
    print('\nreq:' + req + '\n')
    result = requests.get(req, headers=headers)
    print(result.text)

# http://www.giantbomb.com/api/games/?api_key=a0ba8e3ae03028e28fdae2671699af48caf2aeb6&format=json&field_list=name&filter=platforms:1

def get_features(name):

    genres = []

    themes = []

    review_keywords = []

    features = genres + themes + review_keywords

    return features

def get_top_games(platforms):

    names = []
    for platform in platforms:
        names_this_platform = []
        names = names + names_this_platform

    return names

if __name__ == '__main__':
    get_platforms()
