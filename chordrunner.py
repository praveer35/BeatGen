import os
import pty

pid = os.fork()
os.execvp("g++", ["g++", "-std=c++11", "MusicGenerator.cpp"]) if pid == 0 else os.waitpid(pid, 0)

tmpout = os.dup(pty.STDOUT_FILENO)
pipefd = os.pipe()
os.dup2(pipefd[1], pty.STDOUT_FILENO)

if not os.fork(): os.execvp("./a.out", ["./a.out"])

chordStr = os.read(pipefd[0], 1024).decode('UTF-8').strip()
os.dup2(tmpout, 1)
os.close(tmpout)

chords = chordStr.split(' ')
for i in range(len(chords)):
    chords[i] = int(chords[i])

print(chords)








"""
pid = os.fork()
os.execvp("g++", ["g++", "-std=c++11", "MusicGenerator.cpp"]) if pid == 0 else os.waitpid(pid, 0)

tmpout = os.dup(pty.STDOUT_FILENO)
pipefd = os.pipe()
os.dup2(pipefd[1], pty.STDOUT_FILENO)

if not os.fork(): os.execvp("./a.out", ["./a.out"])

chordStr = os.read(pipefd[0], 1024).decode('UTF-8').strip()
os.dup2(tmpout, 1)
os.close(tmpout)

chords = chordStr.split(' ')
for i in range(len(chords)):
    chords[i] = int(chords[i])"""