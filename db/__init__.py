from sqlalchemy import create_engine
import config

# mysql db
user_nm = config.MYSQL_KEY["USER_NM"]
user_pw = config.MYSQL_KEY["USER_PW"]

host_nm = "127.0.0.1:3306"
engine = create_engine("mysql+pymysql://"+user_nm+":"+user_pw+"@"+host_nm, encoding="utf-8")

conn = engine.connect()
