# -*- coding: utf-8 -*-


"""This filters the bookcrossing data
http://www.informatik.uni-freiburg.de/~cziegler/BX/
prerequisite: the data is in the folder /BX-SQL-Dump
excludes all implicit ratings
excludes all users having < 20 ratings
writes the dump to .dat files (same format as in the MovieLens files)
To use this, import the Bookcrossing data set into MySQL first
mysql -u root -p books < BX-Users.sql
mysql -u root -p books < BX-Books.sql
mysql -u root -p books < BX-Book-Ratings.sql
Before doing this, add 'SET autocommit=0;' to the beginnen and 'COMMIT;' to the 
end of the files to massively speed up importing.

Should this not work, try importing the CSV dump as follows:
LOAD DATA INFILE "C:\\PhD\\...\\BX-SQL-Dump\\test.csv" IGNORE
INTO TABLE `bx-books`
COLUMNS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(`ISBN`, `Book-Title`, `Book-Author`, `Year-of-Publication`, @dummy, @dummy, @dummy, @dummy);
"""


from __future__ import division #, unicode_literals
import io
import MySQLdb as mdb
import pdb
import random
import sys


def extract_from_db():
    """do some preprocessing on the database
    and write it to MovieLens style dataset files
    """
    conn = mdb.connect('localhost', 'root', 'master', 'books')
    cursor = conn.cursor()

    # ignore implicit ratings
    stmt = """DELETE FROM ratings WHERE `Book-Rating` = 0"""
    cursor.execute(stmt)
    conn.commit()

    # users
    stmt = """SELECT DISTINCT `User-ID` FROM ratings
              WHERE `User-ID` IN
                  (SELECT `User-ID` FROM ratings
                   GROUP BY `User-ID` HAVING COUNT(`User-ID`) >= 5)
            """
    cursor.execute(stmt)
    users = cursor.fetchall()
    users = set([u[0] for u in users])
    print len(users), 'users'

    # books
    stmt = """SELECT DISTINCT `ISBN` FROM ratings
            WHERE `User-ID` IN (%s)
            AND `ISBN` IN 
                (SELECT `ISBN` FROM ratings
                 GROUP BY `ISBN` HAVING COUNT(`ISBN`) >= 20);
            """ % ', '.join([str(u) for u in users])
    cursor.execute(stmt)
    books = cursor.fetchall()
    books = set([u[0] for u in books])
    books = ["'" + b + "'" for b in books]
    
    stmt = """SELECT `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication` FROM books
            WHERE `ISBN` in (%s)""" % ', '.join([b for b in books])
    cursor.execute(stmt)
    books = cursor.fetchall()
    print len(books), 'books'

    # ratings
    stmt = """SELECT * FROM ratings
            WHERE `User-ID` in (%s)""" % ', '.join([str(u) for u in users])
    cursor.execute(stmt)
    ratings = cursor.fetchall()
    print len(ratings), 'ratings'

    # write to files
    with open('users.dat', 'w') as outfile:
        for u in sorted(users):
            outfile.write(str(u) + '\n')

    with open('books_full.dat', 'w') as outfile:
        for i, t, a, y in sorted(books):
            #outfile.write(i + '::' + t + '\n')
            outfile.write(i + '::' + t + ' (' + str(y) + ')::' + a + '\n')

    with open('ratings.dat', 'w') as outfile:
        for u, i, r in ratings:
            outfile.write(str(u) + '::' + i + '::' + str(r) + '\n')

            
def extract_random_sample(n, exclude_fnames):
    """ extract a random sample from the full data file
    may exclude books from other sampled files (exclude_fnames)"""
    def get_id2line(fname):
        id2line = {}
        with open(fname) as infile:
            for line in infile:
                if not line.strip():
                    continue
                id2line[line.split('::')[0]] = line
        return id2line
    
    full_id2line = get_id2line('books_full.dat')
    
    feid2line = {}
    for fe in exclude_fnames:
        id2line = get_id2line(fe)
        if set(feid2line.keys()) & set(id2line.keys()):
            print "Error - shouldn't get here"
        for k, v in id2line.items():
            feid2line[k] = v
    id2line = {k: v for k, v in full_id2line.items() if not k in feid2line}
             
    ids = random.sample(id2line.keys(), n)
    with io.open('books.dat', 'w', encoding='utf-8') as outfile:
        for i in sorted(ids):
            outfile.write(id2line[i].decode('ISO-8859-1'))
            
            
def get_titles():
    # titles = []
    # with open('books.dat') as infile, open('titles_books.txt', 'w') as outfile:
        # for line in infile:
            # t = line.split('::')[1].rsplit('(', 1)[0]
            # outfile.write(t + '\n')
            
    titlesdb, titlesfiltered = set(), set()
    for line in open('titles_books.txt'):
        titlesfiltered.add(line.strip())
    for line in open('titles_db.txt'):
        titlesdb.add(line.strip())     
    td = titlesdb
    tf = titlesfiltered
    pdb.set_trace()        
            
if __name__ == '__main__':
    # extract_from_db()
    # extract_random_sample(60000)
    get_titles()