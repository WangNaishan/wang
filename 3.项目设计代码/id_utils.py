# -*- coding: utf-8 -*-


"""
存放公用的账户读写函数
"""
import csv
import pymysql

# 写入账户信息到mysql数据库
def sava_id_info_mysql(user, pwd):
    USER_PWD = {}
    conn = pymysql.connect(host='localhost', user='root', password="123456", port=3306, db='ad_database',
                           charset="utf8")
    cur = conn.cursor()
    sql = """INSERT INTO `user` (username, password) VALUES ('{}', '{}')""".format(user, pwd)
    print(sql)
    try:
        # 执行sql语句
        cur.execute(sql)
        # 提交到数据库执行
        conn.commit()
        print("Data inserted successfully.")
    except Exception as e:
        print("Error: unable to insert data -", e)
        # 如果发生错误则回滚
        conn.rollback()
    cur.close()
    conn.close()

# 从mysql数据库中获取账号信息
def get_id_info_mysql():
    """从数据库获取账户信息"""
    USER_PWD = {}
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='ad_database',
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT username, password FROM user")
        result = cursor.fetchall()
        for row in result:
            USER_PWD[row[0]] = row[1]
    except pymysql.MySQLError as e:
        print("数据库操作失败:", e)
    finally:
        if conn:
            conn.close()
    return USER_PWD
