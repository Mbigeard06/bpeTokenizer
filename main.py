from basicTokenizer import BasicTokenizer
from regexTokenizer import RegexTokenizer
import regex as re
import tiktoken

with open("tests/tailorswift.txt", "r", encoding="utf-8") as file:
    text = file.read()  # Read all content into a string


regexTokenizer = RegexTokenizer()
regexTokenizer.train(text, 351)

enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("hello world!!!? (안녕하세요!) lol123 😉")
text = enc.decode(ids) # get the same text back
#test to see if our logic matches tiktoken 
if "hello world!!!? (안녕하세요!) lol123 😉" == regexTokenizer.decode(regexTokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉")):
    print("encoding successful")