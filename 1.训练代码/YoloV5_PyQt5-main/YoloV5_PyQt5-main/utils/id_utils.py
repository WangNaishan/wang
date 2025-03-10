# -*- coding: utf-8 -*-


'''
存放公用的账户读写函数
'''
import csv

# 写入账户信息到csv文件
def sava_id_info(user, pwd):
    headers = ['name', 'key']
    values = [{'name':user, 'key':pwd}]
    with open('userInfo.csv', 'a', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, headers)
        writer.writerows(values)

# 写入账户信息到mysql数据库
def sava_id_info_mysql(user, pwd):
    USER_PWD = {}
    conn = pymysql.connect(host='localhost', user='root', passwd="123456", port=3306, db='demo', charset="utf8")
    cur = conn.cursor()
    sql = """insert into `user` VALUES('{}','{}')""".format(user,pwd)
    print(sql)
    try:
        # 执行sql语句
        cur.execute(sql)
        # 提交到数据库执行
        conn.commit()
        # 显示出所有数据
        result = cur.fetchall()
        if result:
            for r in result:
                USER_PWD[r[0]]=r[1]
                # print(r)
        else:
            print("error!")
    except:
        print("Error: unable to insert data")
        # 如果发生错误则回滚
        conn.rollback()
    cur.close()
    conn.close()

# 读取csv文件获得账户信息
def get_id_info():
    USER_PWD = {}
    with open('userInfo.csv', 'r') as csvfile: # 此目录即是当前项目根目录
        spamreader = csv.reader(csvfile)
        # 逐行遍历csv文件,按照字典存储用户名与密码
        for row in spamreader:
            USER_PWD[row[0]] = row[1]
    return USER_PWD

import pymysql
# 从mysql数据库中获取账号信息
def get_id_info_mysql():
    USER_PWD = {}
    conn = pymysql.connect(host='localhost', user='root', passwd="123456", port=3306, db='demo', charset="utf8")
    cur = conn.cursor()
    sql = """select * from user"""
    try:
        # 执行sql语句
        cur.execute(sql)
        # 提交到数据库执行
        conn.commit()
        # 显示出所有数据
        result = cur.fetchall()
        if result:
            for r in result:
                USER_PWD[r[0]]=r[1]
                # print(r)
        else:
            print("error!")
    except:
        print("Error: unable to fetch data")
        # 如果发生错误则回滚
        conn.rollback()
    cur.close()
    conn.close()
    return USER_PWD

# get_id_info_mysql()


