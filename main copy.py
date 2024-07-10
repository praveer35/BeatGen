import sqlite3
from io import BytesIO
import os
import random
import datetime
import hashlib
import time
import string

# add a user to database
def save_tracks(chords, melody, arpeggio):
    conn = sqlite3.connect("Databases/muzeic.db")#check if the username and email are unique (this is purely for user feedback)
    print('reached here')
    cur = conn.cursor()                         #as emails and usernames being unique is also enforced by the db
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sql = f'''
        INSERT INTO Tracks(user_id, timestamp, name, chords,
            melody, arpeggio)
        VALUES(?,?,?,?,?,?)
    '''

    sql = '''
    INSERT INTO UsersTable(user_id, username, pw, 
        joined, last_accessed, num_accessed, email, posts, projects, comments_liked)
    VALUES(?,?,?,?,?,?,?,?,?,?)
    '''

    try:
        cur.execute(sql, (1601, now, 'praveer', chords.join(' '), melody.join(' '), arpeggio.join(' ')))
        conn.commit()
    except sqlite3.IntegrityError:
        # if by coincidence, we try to add another user id that already exists, try again
        cur.close()
        conn.close()  

    cur.close()
    conn.close()

    return 'done'

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData