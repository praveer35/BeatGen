# import sqlite3

# conn = sqlite3.connect('users.db')
# cursor = conn.cursor()

import os

soundfount_titles = [x[:-4] for x in os.listdir("Soundfonts")]
print(soundfount_titles)