import sqlite3

conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# sql = "SELECT generation_id, track_name, track_data FROM generations"
# cursor.execute(sql)
# x = cursor.fetchall()

#print(x)

# for n in x:
#     print(n[0], n[1])
#     cursor.execute("INSERT INTO generation_data (generation_id, track_name, track_data) VALUES (?, ?, ?)",
#         (n[0], n[1], n[2]))
#     conn.commit()

generation_ids = set()

sql = "SELECT * FROM generations"
cursor.execute(sql)
x = cursor.fetchall()
for n in x:
    if n[2] not in generation_ids:
        generation_ids.add(n[2])
        cursor.execute("INSERT INTO new_generations (user_id, created, generation_id) VALUES (?, ?, ?)",
            (n[0], n[1], n[2]))
        conn.commit()