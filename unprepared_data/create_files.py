import os
import json

parent_dir = "data"
if not os.path.isdir(parent_dir):
    os.mkdir(parent_dir)

# futurama
futurama_quotes = "futurama-quotes.php"
futurama_dir = os.path.join(parent_dir, "futurama")

if not os.path.isdir(futurama_dir):
    os.mkdir(futurama_dir)
    with open(futurama_quotes,"r") as f:
        i = 1
        for line in f:
            file_name = str(i) + ".txt"
            with open(os.path.join(futurama_dir, file_name),"x") as df:
                df.write(line.split(": ", 1)[1])
            i = i + 1

# nietzche
nietzche_quotes = "nietzche-quotes.txt"
nietzche_dir = os.path.join(parent_dir, "nietzche")

if not os.path.isdir(nietzche_dir):
    os.mkdir(nietzche_dir)
    with open(nietzche_quotes,"r") as f:
        i = 1
        for line in f:
            file_name = str(i) + ".txt"
            with open(os.path.join(nietzche_dir, file_name),"x") as df:
                df.write(line)
            i = i + 1

# stoics
stoic_quotes = "stoic-quotes.json"
stoic_dir = os.path.join(parent_dir, "stoic")

if not os.path.isdir(stoic_dir):
    os.mkdir(stoic_dir)
    with open(stoic_quotes,"r") as f:
        quotes = json.load(f)
        for i, quote in enumerate(quotes["quotes"]):
            file_name = str(i) + ".txt"
            with open(os.path.join(stoic_dir, file_name),"x") as df:
                df.write(quote['text'])
