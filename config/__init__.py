import pathlib
import json

PATH_MYSQL_KEY = pathlib.Path(__file__).parents[1] / 'config' / f'mysql_key.json'

with open(PATH_MYSQL_KEY, 'r', encoding='utf-8') as fp:
    MYSQL_KEY = json.load(fp)
