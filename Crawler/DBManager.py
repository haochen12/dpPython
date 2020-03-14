import MySQLdb as mysqldb


class DBManager(object):
    def __init__(self):
        self.db = mysqldb.connect("localhost", "root", "123456", "test", charset='utf8')
        self.cursor = self.db.cursor()

    def get_version(self):

        # 使用execute方法执行SQL语句
        self.cursor.execute("SELECT VERSION()")

        # 使用 fetchone() 方法获取一条数据
        data = self.cursor.fetchone()

        print("Database version : %s " % data)

    def insert_data(self):
        insert_sql = "INSERT INTO douban_commits (id, commits_text, commits_score, commits_use_count," \
                     " commits_user_name, commits_date) VALUES (%s,%s,%s,%s,%s, %s)" % \
                     ("'sfs'", "'sdfs'", "'sdsdf'", "'sdsdf'", "'fsfdf'", "'7987'")
        try:
            self.cursor.execute(insert_sql)
            self.db.commit()
        except:
            print("insert error!")

    def insert_data1(self, param1, param2, param3, param4, param5):
        insert_sql = "INSERT INTO douban_commits (id, commits_text, " \
                     "commits_score, " \
                     "commits_use_count," \
                     "commits_user_name," \
                     "commits_date) VALUES " \
                     "('%s','%s','%s','%s','%s', '%s')" % (str(param1),
                                                           str(param1),
                                                           str(param2),
                                                           str(param3),
                                                           str(param4),
                                                           str(param5))
        print(insert_sql)
        try:
            self.cursor.execute(insert_sql)
            self.db.commit()
        except:
            print("insert error!")

    def select_data(self):
        select_sql = "select * from douban_commits order by commits_date"
        try:
            self.cursor.execute(select_sql)
            results = self.cursor.fetchall()
            return results
        except Exception:
            print("select error!")

    def select_data_by_date(self):
        select_sql = "select commits_date from douban_commits order by commits_date"
        try:
            self.cursor.execute(select_sql)
            results = self.cursor.fetchall()
            for row in results:
                print(row)
        except Exception:
            print("select error!")

    def delete_data(self):
        delete_sql = "delete from douban_commits where id like 'sfs' "
        try:
            self.cursor.execute(delete_sql)
            self.db.commit()
        except Exception:
            print("delete error")

    def update_data(self):
        update_sql = "update douban_commits set commits_text = 'test' where id like '111s'"
        try:
            self.cursor.execute(update_sql)
            self.db.commit()
        except:
            print("update error")

    def __del__(self):
        # 关闭数据库连接
        self.db.close()


if __name__ == "__main__":
    db_test = DBManager()
    db_test.select_data_by_date()
