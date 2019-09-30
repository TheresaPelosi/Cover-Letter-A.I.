import glob

path = "C:\\Users\mewlo\Downloads\cover_letter_examples___Google_Search"

files = [f for f in glob.glob(path + "**/*", recursive=True)]

for f in files:
    print(f)