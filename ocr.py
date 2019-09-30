import pytesseract

path = "C:\\Users\mewlo\Downloads\cover_letter_examples___Google_Search\\batch.txt"

text = pytesseract.image_to_string(path)

with open("input.txt", 'a+', encoding='utf-8') as f:
    f.write(text)